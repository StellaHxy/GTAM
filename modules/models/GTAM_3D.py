import os
import os.path as osp
import warnings
from math import pi as PI

import ase
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding, Linear, ModuleList, Sequential
from torch_geometric.data.makedirs import makedirs
from torch_geometric.nn import MessagePassing, radius_graph
from torch_geometric.utils import degree, softmax, to_dense_batch, to_dense_adj
from torch_scatter import scatter
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims 
from einops import rearrange


class PairLoss3Dto2D(nn.Module):
    def __init__(self, input_size, output_size=6):
        super(PairLoss3Dto2D, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        return x


class GTAM_3D(torch.nn.Module):
    def __init__(
        self,
        evo_config=None,
        num_bins=32,
        hidden_channels=128,
        num_filters=128,
        num_interactions=6,
        num_gaussians=50,
        cutoff=10.0,
        node_class=None,
        readout="mean",
        dipole=False,
        mean=None,
        std=None,
        atomref=None,
    ):
        super(SchNet, self).__init__()

        assert readout in ["add", "sum", "mean"]

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.readout = readout
        self.dipole = dipole
        self.readout = "add" if self.dipole else self.readout
        self.mean = mean
        self.std = std
        self.scale = None
        self.evo_config = evo_config

        atomic_mass = torch.from_numpy(ase.data.atomic_masses)
        self.register_buffer("atomic_mass", atomic_mass)

        # self.embedding = Embedding(100, hidden_channels)
        self.embedding = Embedding(node_class, hidden_channels)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)
        self.zij_distance = GaussianSmearing(0.0, cutoff, 64)

        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(
                hidden_channels, num_gaussians, num_filters, cutoff
            )
            self.interactions.append(block)

        # TODO: double-check hidden size
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.act = ShiftedSoftplus()
        self.lin2 = Linear(hidden_channels, hidden_channels)

        self.register_buffer("initial_atomref", atomref)
        self.atomref = None
        if atomref is not None:
            self.atomref = Embedding(100, 1)
            self.atomref.weight.data.copy_(atomref)
        
        self.evo_config = evo_config
        if evo_config != None:
            self.evo_emb_dim = 64
            self.seqformer = Seqformer(evo_config)
            self.zij_embedding = torch.nn.Embedding(num_bins, self.evo_emb_dim, padding_idx=0)
            torch.nn.init.xavier_uniform_(self.zij_embedding.weight.data)
            
            self.pair_3Dto2D_loss = PairLoss3Dto2D(input_size=self.evo_emb_dim, output_size=6)
            
        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)
        if self.atomref is not None:
            self.atomref.weight.data.copy_(self.initial_atomref)

    def get_distance(self, positions, args):
        breaks = torch.linspace(args.first_break, args.last_break, steps=args.num_bins-1).to(positions.device)

        sq_breaks = torch.square(breaks)

        dist2 = torch.sum(
            torch.square(
                rearrange(positions,'b l c -> b l () c') - 
                rearrange(positions, 'b l c -> b () l c')),
                dim = -1,
                keepdims = True)

        true_bins = torch.sum(dist2 > sq_breaks, dim=-1)

        return true_bins

    def get_node_representation(self, seq_act, data):
        node_representation = None
        for i in range(data.batch_size):
            if node_representation == None: 
                node_representation = seq_act[i][:data.atom_num[i]]
            else:
                node_representation = torch.cat((node_representation, seq_act[i][:data.atom_num[i]]), 0)
        return node_representation

    def forward(self, data, return_latent=False, is_finetune=False):
        # if data.x.dim()==3:
        #     z = data.x[:, 0]
        # else:
        #     z = data.x
        z = data.x[:, 0]  
        pos = data.positions
        batch = data.batch
        assert z.dim() == 1 and z.dtype == torch.long
        batch = torch.zeros_like(z) if batch is None else batch

        h = self.embedding(z)

        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        edge_attr = self.distance_expansion(edge_weight)
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)

        if self.dipole:
            # Get center of mass.
            mass = self.atomic_mass[z].view(-1, 1)
            c = scatter(mass * pos, batch, dim=0) / scatter(mass, batch, dim=0)
            h = h * (pos - c[batch])

        if not self.dipole and self.mean is not None and self.std is not None:
            h = h * self.std + self.mean

        if not self.dipole and self.atomref is not None:
            h = h + self.atomref(z)
            
        # TODO: zij, Evoformer
        
        if self.evo_config != None:
            h, mask = to_dense_batch(h, data.atom_batch)
            
            pair_mask = to_dense_adj(data.edge_index, data.atom_batch).unsqueeze(-1)
            pair_mask_3D = to_dense_adj(edge_index, data.atom_batch).unsqueeze(-1)

            zij = self.zij_distance(edge_weight)
            h_ij = to_dense_adj(edge_index, data.atom_batch, zij)

            seq_act, pair_act = self.seqformer(h, h_ij, mask, pair_mask, is_recycling=False)
            h = self.get_node_representation(seq_act, data)

            zij = self.pair_3Dto2D_loss(pair_act)

        out = scatter(h, batch, dim=0, reduce=self.readout)

        if is_finetune:
            node_representation = scatter(h, data.batch, dim=0, reduce="add")
            if self.dipole:
                node_representation = torch.norm(node_representation, dim=-1, keepdim=True)
            if self.scale is not None:
                node_representation = self.scale * node_representation
            return node_representation 
        
        if self.dipole:
            out = torch.norm(out, dim=-1, keepdim=True)
        if self.scale is not None:
            out = self.scale * out
        if return_latent:
            return out, h, zij, pair_mask_3D
        
        return out, zij, pair_mask_3D

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"hidden_channels={self.hidden_channels}, "
            f"num_filters={self.num_filters}, "
            f"num_interactions={self.num_interactions}, "
            f"num_gaussians={self.num_gaussians}, "
            f"cutoff={self.cutoff})"
        )


class InteractionBlock(torch.nn.Module):
    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff):
        super(InteractionBlock, self).__init__()
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.conv = CFConv(
            hidden_channels, hidden_channels, num_filters, self.mlp, cutoff
        )
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[0].bias.data.fill_(0)
        self.conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class CFConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_filters, nn, cutoff):
        super(CFConv, self).__init__(aggr="add")
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = nn
        self.cutoff = cutoff

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        C = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)
        W = self.nn(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j, W):
        return x_j * W


class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift
