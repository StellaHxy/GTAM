
import os
import pandas as pd
import numpy as np
from itertools import repeat
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import subgraph, to_networkx, remove_self_loops
from torch_sparse import coalesce, spspmm
import os.path as osp


def extend_graph(data):
    edge_index = data.edge_index
    N = data.num_nodes

    value = edge_index.new_ones((edge_index.size(1),), dtype=torch.float)

    index, value = spspmm(edge_index, value, edge_index, value, N, N, N)
    value.fill_(0)
    index, value = remove_self_loops(index, value)

    edge_index = torch.cat([edge_index, index], dim=1)

    edge_index, _ = coalesce(edge_index, None, N, N)

    value = edge_index.new_ones((edge_index.size(1),), dtype=torch.float)

    index, value = spspmm(edge_index, value, edge_index, value, N, N, N)
    value.fill_(0)
    index, value = remove_self_loops(index, value)

    edge_index = torch.cat([edge_index, index], dim=1)

    data.extended_edge_index, _ = coalesce(edge_index, None, N, N)
    return data


class Molecule3DDataset(InMemoryDataset):
    def __init__(self, root, dataset, mask_ratio=0, remove_center=False, transform=None, pre_transform=None, pre_filter=None, empty=False, use_extend_graph=False):
        self.root = root
        self.dataset = dataset
        self.mask_ratio = mask_ratio
        self.remove_center = remove_center
        self.use_extend_graph = use_extend_graph

        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
        super(Molecule3DDataset, self).__init__(root, transform, pre_transform, pre_filter)

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])
            self.edge_attr_dim = int(self.data.edge_attr.shape[1])
            self.nodes_index = self.slices['x']
            self.edges_index = self.slices['edge_index']
        print('Dataset: {}\nData: {}'.format(self.dataset, self.data))


    def subgraph(self, data):
        G = to_networkx(data)
        node_num = data.x.size()[0]
        sub_num = int(node_num * (1 - self.mask_ratio))

        idx_sub = [np.random.randint(node_num, size=1)[0]]
        idx_neigh = set([n for n in G.neighbors(idx_sub[-1])])

        # BFS
        while len(idx_sub) <= sub_num:
            if len(idx_neigh) == 0:
                idx_unsub = list(set([n for n in range(node_num)]).difference(set(idx_sub)))
                idx_neigh = set([np.random.choice(idx_unsub)])
            sample_node = np.random.choice(list(idx_neigh))

            idx_sub.append(sample_node)
            idx_neigh = idx_neigh.union(
                set([n for n in G.neighbors(idx_sub[-1])])).difference(set(idx_sub))

        idx_nondrop = idx_sub
        idx_nondrop.sort()

        edge_idx, edge_attr = subgraph(
            subset=idx_nondrop,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            relabel_nodes=True,
            num_nodes=node_num
        )
        data.edge_index = edge_idx
        data.edge_attr = edge_attr
        data.x = data.x[idx_nondrop]
        data.positions = data.positions[idx_nondrop]
        data.__num_nodes__ = data.x.size()[0]
        
        if "radius_edge_index" in data:
            radius_edge_index, _ = subgraph(
                subset=idx_nondrop,
                edge_index=data.radius_edge_index,
                relabel_nodes=True, 
                num_nodes=node_num)
            data.radius_edge_index = radius_edge_index
        if "extended_edge_index" in data:
            # TODO: may consider extended_edge_attr
            extended_edge_index, _ = subgraph(
                subset=idx_nondrop,
                edge_index=data.extended_edge_index,
                relabel_nodes=True, 
                num_nodes=node_num)
            data.extended_edge_index = extended_edge_index
        # TODO: will also need to do this for other edge_index
        return data


    def get(self, idx):
        data = Data()
        for key in self.data.keys():
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]

        if self.use_extend_graph:
            data = extend_graph(data)

        if self.mask_ratio > 0:
            data = self.subgraph(data)

        if self.remove_center:
            center = data.positions.mean(dim=0)
            data.positions -= center
        data.atom_num = int(self.nodes_index[idx+1] - self.nodes_index[idx])
        data.edge_num = int(self.edges_index[idx+1]) - int(self.edges_index[idx])

        return data
    
    def _download(self):
        return

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def process(self):
        return
    
    def get_idx_split(self):
        split_dict = torch.load(osp.join(self.root, 'split_dict.pt'))
        return split_dict
