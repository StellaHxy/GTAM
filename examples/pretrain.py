import sys
sys.path.append(r"..")

import time
import os
import json
import numpy as np
import torch
import ml_collections
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

from modules.datasets import Molecule3DDataset
from modules.models import GTAM_2D, GTAM_3D
from modules.models.SDE import SDEModel2Dto3D_02, SDEModel3Dto2D_node_adj_dense
from util import dual_CL, do_2dTo3d_loss, do_3dTo2d_loss
from config import args
from ogb.utils.features import get_atom_feature_dims

ogb_feat_dim = get_atom_feature_dims()
ogb_feat_dim = [x - 1 for x in ogb_feat_dim]
ogb_feat_dim[-2] = 0
ogb_feat_dim[-1] = 0

global epoch
epoch=0


def save_model(save_best):
    global epoch
    epoch += 1
    if not args.output_model_dir == '':
        if save_best:
            global optimal_loss
            print('save model with loss: {:.5f}'.format(optimal_loss))
            output_model_path = os.path.join(args.output_model_dir, f"model_complete_{epoch}.pth")
            saver_dict = {
                'model_2D': molecule_model_2D.state_dict(),
                'model_3D': molecule_model_3D.state_dict(),
                'SDE_2Dto3D_model': SDE_2Dto3D_model.state_dict(),
                'SDE_3Dto2D_model': SDE_3Dto2D_model.state_dict(),
            }
            torch.save(saver_dict, output_model_path)

        else:
            output_model_path = os.path.join(args.output_model_dir, "model_complete_final.pth")
            saver_dict = {
                'model_2D': molecule_model_2D.state_dict(),
                'model_3D': molecule_model_3D.state_dict(),
                'SDE_2Dto3D_model': SDE_2Dto3D_model.state_dict(),
                'SDE_3Dto2D_model': SDE_3Dto2D_model.state_dict(),
            }
            torch.save(saver_dict, output_model_path)
    return


def get_atom_batch(batch):
    atom_batch = []
    for i in range(batch.batch_size):
        atom_batch += [i for _ in range(batch.atom_num[i])]
    return torch.tensor(atom_batch)


def count_parameters(model):
    count = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            count.append(param)
            print(f"name: {name}")
    sum_para = sum(p.numel() for p in count)
    print(f"Total trainable parameters: {sum_para}")
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(args, molecule_model_2D, device, loader, optimizer):
    start_time = time.time()

    molecule_model_2D.train()
    molecule_model_3D.train()
    SDE_2Dto3D_model.train()
    SDE_3Dto2D_model.train()

    # if args.SDE_coeff_2D_masking > 0:
    #     molecule_atom_masking_model.train()
    # if args.SDE_coeff_3D_masking > 0:
    #     GeoSSL_moel.train()

    SDE_loss_2Dto3D_accum, SDE_loss_3Dto2D_accum = 0, 0
    CL_loss_accum, CL_acc_accum = 0, 0
    pair_2Dto3D_loss_accum, pair_3Dto2D_loss_accum = 0, 0
    
    if args.verbose:
        l = tqdm(loader)
    else:
        l = loader
    for step, batch in enumerate(l):
        
        batch.atom_batch = get_atom_batch(batch)
        batch = batch.to(device)

        node_2D_repr, zij_2d, pair_mask = molecule_model_2D(batch)
        _, node_3D_repr, zij_3d, pair_mask_3D = molecule_model_3D(batch, return_latent=True)
        
        loss = 0

        # TODO: +loss
        if args.Pair_2Dto3D_coeff > 0:
            pair_2Dto3D_loss = do_2dTo3d_loss(zij_2d, batch, pair_mask, args)
            loss += args.Pair_2Dto3D_coeff * pair_2Dto3D_loss
            print(f"loss: {pair_2Dto3D_loss}")
            pair_2Dto3D_loss_accum += args.Pair_2Dto3D_coeff * pair_2Dto3D_loss.detach().cpu().item()
            print(f"Loss/train@pair_2Dto3D_loss: {args.Pair_2Dto3D_coeff * pair_2Dto3D_loss.detach().cpu().item()}")
        
        if args.Pair_3Dto2D_coeff > 0:
            pair_3Dto2D_loss = do_3dTo2d_loss(zij_3d, batch, args)
            loss += args.Pair_3Dto2D_coeff * pair_3Dto2D_loss
            pair_3Dto2D_loss_accum += args.Pair_3Dto2D_coeff * pair_3Dto2D_loss.detach().cpu().item()
            print(f"Loss/train@pair_3Dto2D_loss: {args.Pair_3Dto2D_coeff * pair_3Dto2D_loss.detach().cpu().item()}")
        
        if args.SDE_coeff_contrastive > 0:
            CL_loss, CL_acc = dual_CL(node_2D_repr, node_3D_repr, args)
            loss += CL_loss * args.SDE_coeff_contrastive
            CL_loss_accum += CL_loss.detach().cpu().item()
            CL_acc_accum += CL_acc
            print(f"Loss/train@CL_loss: {args.SDE_coeff_contrastive * CL_loss.detach().cpu().item()}")

        if args.SDE_coeff_generative_2Dto3D > 0:
            SDE_loss_2Dto3D_result = SDE_2Dto3D_model(node_2D_repr, batch, anneal_power=args.SDE_anneal_power)
            SDE_loss_2Dto3D = SDE_loss_2Dto3D_result["position"]
            loss += SDE_loss_2Dto3D * args.SDE_coeff_generative_2Dto3D
            SDE_loss_2Dto3D_accum += SDE_loss_2Dto3D.detach().cpu().item()
            print(f"Loss/train@SDE_loss_2Dto3D: {args.SDE_coeff_generative_2Dto3D * SDE_loss_2Dto3D.detach().cpu().item()}")
        
        if args.SDE_coeff_generative_3Dto2D > 0:
            SDE_loss_3Dto2Dx, SDE_loss_3Dto2Dadj = SDE_3Dto2D_model(node_3D_repr, batch, reduce_mean=reduce_mean, continuous=True, train=True, anneal_power=args.SDE_anneal_power)
            SDE_loss_3Dto2D = (SDE_loss_3Dto2Dx + SDE_loss_3Dto2Dadj) * 0.5
            loss += SDE_loss_3Dto2D * args.SDE_coeff_generative_3Dto2D
            SDE_loss_3Dto2D_accum += SDE_loss_3Dto2D.detach().cpu().item()
            print(f"Loss/train@SDE_loss_3Dto2D: {args.SDE_coeff_generative_3Dto2D * SDE_loss_3Dto2D.detach().cpu().item()}")  
        
        print(f"Loss/train@all_loss: {loss.detach().cpu().item()}") 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    global optimal_loss
    CL_loss_accum /= len(loader)
    CL_acc_accum /= len(loader)
    SDE_loss_2Dto3D_accum /= len(loader)
    SDE_loss_3Dto2D_accum /= len(loader)
    pair_2Dto3D_loss_accum /= len(loader)
    pair_3Dto2D_loss_accum /= len(loader)
    
    temp_loss = \
        args.SDE_coeff_contrastive * CL_loss_accum + \
        args.SDE_coeff_generative_2Dto3D * SDE_loss_2Dto3D_accum + \
        args.SDE_coeff_generative_3Dto2D * SDE_loss_3Dto2D_accum + \
        args.Pair_2Dto3D_coeff * pair_2Dto3D_loss_accum + \
        args.Pair_3Dto2D_coeff * pair_3Dto2D_loss_accum
    if temp_loss < optimal_loss:
        optimal_loss = temp_loss
        save_model(save_best=True)
        
    print('CL Loss: {:.5f}\tCL Acc: {:.5f}\t\tSDE 2Dto3D Loss: {:.5f}\tSDE 3Dto2D Loss: {:.5f}\t Pair 2Dto3D Loss:{:.5f}\t Pair 3Dto2D Loss:{:.5f}'.format(
        CL_loss_accum, CL_acc_accum, SDE_loss_2Dto3D_accum, SDE_loss_3Dto2D_accum, pair_2Dto3D_loss_accum, pair_3Dto2D_loss_accum))
    print('Time: {:.5f}\n'.format(time.time() - start_time))
    return


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device('cuda:' + str(args.device)) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.cuda.set_device(args.device)
    node_class = 119

    transform = None
    data_root = "{}/{}".format(args.input_data_dir, args.dataset)
    dataset = Molecule3DDataset(data_root, args.dataset, mask_ratio=args.SSL_masking_ratio, remove_center=True, use_extend_graph=args.use_extend_graph, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, prefetch_factor=128)

    # set up model
    with open("../config/GTAformer.json", 'r', encoding='utf-8') as f_evo:
       evo_config = json.loads(f_evo.read())
       evo_config = ml_collections.ConfigDict(evo_config)

    molecule_model_2D = GTAM_2D(
        num_layer = args.num_layer, 
        emb_dim = args.emb_dim, 
        evo_config = evo_config, 
        JK=args.JK, 
        num_bins=args.num_bins , 
        drop_ratio=args.dropout_ratio, 
        gnn_type=args.gnn_type
    ).to(device)
    molecule_readout_func = global_mean_pool

    molecule_model_3D = GTAM_3D(
        evo_config=evo_config,
        hidden_channels=args.emb_dim,
        num_filters=args.SchNet_num_filters,
        num_interactions=args.SchNet_num_interactions,
        num_gaussians=args.SchNet_num_gaussians,
        cutoff=args.SchNet_cutoff,
        readout=args.SchNet_readout,
        node_class=node_class,
    ).to(device)
        
    SDE_2Dto3D_model = SDEModel2Dto3D_02(
                emb_dim=args.emb_dim, hidden_dim=args.hidden_dim_2Dto3D,
                beta_min=args.beta_min_2Dto3D, beta_max=args.beta_max_2Dto3D, 
                num_diffusion_timesteps=args.num_diffusion_timesteps_2Dto3D,
                beta_schedule=None,
                SDE_type=args.SDE_type_2Dto3D,
                use_extend_graph=args.use_extend_graph
    ).to(device)

    if args.noise_on_one_hot:
        reduce_mean = True
    else:
        reduce_mean = False

    SDE_3Dto2D_model = SDEModel3Dto2D_node_adj_dense(
            dim3D=args.emb_dim, c_init=2, c_hid=8, c_final=4, num_heads=4, adim=16,
            nhid=16, num_layers=4,
            emb_dim=args.emb_dim, num_linears=3,
            beta_min=args.beta_min_3Dto2D, beta_max=args.beta_max_3Dto2D, 
            num_diffusion_timesteps=args.num_diffusion_timesteps_3Dto2D,
            SDE_type=args.SDE_type_3Dto2D, num_class_X=node_class, 
            noise_on_one_hot=args.noise_on_one_hot).to(device)
    
    model_param_group = []
    model_param_group.append({'params': molecule_model_2D.parameters(), 'lr': args.lr * args.gnn_2d_lr_scale})
    model_param_group.append({'params': molecule_model_3D.parameters(), 'lr': args.lr * args.gnn_3d_lr_scale})
    model_param_group.append({'params': SDE_2Dto3D_model.parameters(), 'lr': args.lr * args.gnn_2d_lr_scale})
    model_param_group.append({'params': SDE_3Dto2D_model.parameters(), 'lr': args.lr * args.gnn_3d_lr_scale})

    model_2d = count_parameters(molecule_model_2D)
    model_3d = count_parameters(molecule_model_3D)
    tmp_dif_2 = count_parameters(SDE_2Dto3D_model)
    tmp_dif_3 = count_parameters(SDE_3Dto2D_model)
    print("count of parameters", model_2d+model_3d+tmp_dif_2+tmp_dif_3)

    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    optimal_loss = 1e10
    SDE_coeff_contrastive_oriGINal = args.SDE_coeff_contrastive
    args.SDE_coeff_contrastive = 0

    for epoch in range(1, args.epochs + 1):
        if epoch > args.SDE_coeff_contrastive_skip_epochs:
            args.SDE_coeff_contrastive = SDE_coeff_contrastive_oriGINal
        print('epoch: {}'.format(epoch))
        print('device:', device)
        train(args, molecule_model_2D, device, loader, optimizer)

    save_model(save_best=False)
