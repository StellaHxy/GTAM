import os
import torch
import pandas as pd

import numpy as np
from tqdm import tqdm
from rdkit import Chem
import os.path as osp
from itertools import repeat
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from modules.datasets.dataset_utils import mol_to_graph_data_obj_simple_3D
from ogb.utils.features import (atom_to_feature_vector, bond_to_feature_vector)

def smiles2graph(smiles_string):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        # atoms
        atom_features_list = []
        for atom in mol.GetAtoms():
            atom_features_list.append(atom_to_feature_vector(atom))
        x = np.array(atom_features_list, dtype = np.int64)

        # bonds
        num_bond_features = 3  # bond type, bond stereo, is_conjugated
        if len(mol.GetBonds()) > 0: # mol has bonds
            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                edge_feature = bond_to_feature_vector(bond)

                # add edges in both directions
                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)

            # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
            edge_index = np.array(edges_list, dtype = np.int64).T

            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            edge_attr = np.array(edge_features_list, dtype = np.int64)

        else:   # mol has no bonds
            edge_index = np.empty((2, 0), dtype = np.int64)
            edge_attr = np.empty((0, num_bond_features), dtype = np.int64)
        
        graph = dict()
        graph['edge_index'] = edge_index
        graph['edge_feat'] = edge_attr
        graph['node_feat'] = x
        graph['num_nodes'] = len(x)

        return graph
    except:
        return None


class PCQM4Mv2(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.root = root

        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        super(PCQM4Mv2, self).__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])
        print("root: {},\ndata: {}".format(self.root, self.data))
        return

    @property
    def raw_dir(self):
        return os.path.join(self.root, 'raw')

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_file_names(self):
        name = 'pcqm4m-v2-train.sdf '
        return name

    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"

    def download(self):
        pass

    def process(self):
        data_df = pd.read_csv(os.path.join(self.raw_dir, 'data.csv.gz'))
        smiles_list = data_df['smiles']
        homolumogap_list = data_df['homolumogap']

        data_list, data_smiles_list = [], []

        sdf_file = "{}/{}".format(self.raw_dir, self.raw_file_names).strip()

        print('Converting SMILES strings into graphs...')
        for idx, s in tqdm(enumerate(smiles_list)):
            # import pdb
            # pdb.set_trace()
            graph = smiles2graph(s)
            try:
                data = Data()

                assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
                assert (len(graph['node_feat']) == graph['num_nodes'])

                data.__num_nodes__ = int(graph['num_nodes'])
                data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
                data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
                data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
                data.y = torch.Tensor([homolumogap_list[idx]])
                data.positions = torch.zeros(data.__num_nodes__, 3).to(torch.float32)

                data_list.append(data)

            except:
                continue
        # import pdb
        # pdb.set_trace()
        print('Extracting 3D positions from SDF files for Training Data...')
        train_data_with_position_list = []
        suppl = Chem.SDMolSupplier(sdf_file)
        for idx, mol in tqdm(enumerate(suppl)):
            data, _ = mol_to_graph_data_obj_simple_3D(mol)

            data.y = homolumogap_list[idx]
            train_data_with_position_list.append(data)


        data_list = train_data_with_position_list + data_list[len(train_data_with_position_list):]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data_smiles_series = pd.Series(smiles_list)
        saver_path = os.path.join(self.processed_dir, "smiles.csv")
        print("saving to {}".format(saver_path))
        data_smiles_series.to_csv(saver_path, index=False, header=False)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        return

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))

    def get(self, idx):
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys():
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]

        return data

    def get_idx_split(self):
        split_dict = torch.load(osp.join(self.root, 'split_dict.pt'))
        return split_dict