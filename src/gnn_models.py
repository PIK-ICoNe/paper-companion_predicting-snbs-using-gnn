import os
import numpy as np
import h5py

import torch as torch
import torch.nn as nn
import torch.nn.functional as F


from torch_geometric.data import Data as gData
from torch_geometric.data import Dataset as gDataset
from torch_geometric.nn import GCNConv, ARMAConv, SGConv, TAGConv


def get_length_of_dataset(grid_path):
    count = 0
    for file in sorted(os.listdir(grid_path)):
        if file.startswith('grid_data_'):
            if count == 0:
                startIndex = int(os.path.splitext(
                    file)[0].split('grid_data_')[1])
                digits = (os.path.splitext(
                    file)[0].split('grid_data_')[1])
            count += 1
    return count, startIndex, digits


class gnn_snbs(gDataset):
    def __init__(self, grid_path, slice_index=slice(0, 0)):
        if slice_index.stop == 0:
            self.data_len, self.start_index, digits = get_length_of_dataset(
                grid_path)
        else:
            _, _, digits = get_length_of_dataset(grid_path)
            self.start_index = slice_index.start + 1
            self.data_len = slice_index.stop - slice_index.start + 1

        self.path = grid_path
        self.num_digits = '0' + str(digits.__str__().__len__())
        self.num_classes = 1

    def __len__(self):
        return self.data_len

    def num_classes(self):
        return self.num_classes

    def __get_input__(self, index):
        if index + 1 > self.data_len:
            print('Error in dataset: Trying to access invalid element')
        else:
            id = format(index+self.start_index, self.num_digits)
            file_to_read = str(self.path)+'/grid_data_'+str(id) + '.h5'
            hf = h5py.File(file_to_read, 'r')
            # read in sources/sinks
            dataset_P = hf.get('P')
            P = np.array(dataset_P)
            # read in edge_index
            dataset_edge_index = hf.get('edge_index')
            edge_index = np.array(dataset_edge_index)-1
            # read in edge_attr
            dataset_edge_attr = hf.get('edge_attr')
            edge_attr = np.array(dataset_edge_attr)

            hf.close()
        return torch.tensor(P).unsqueeze(0).transpose(0, 1), torch.tensor(edge_index).transpose(1, 0), torch.tensor(edge_attr)

    def __get_P__(self, index):
        P, _, _ = self.__get_input__(index)
        return P

    def __get_edge_index__(self, index):
        _, edge_index, _ = self.__get_input__(index)
        return edge_index

    def __get_edge_attr__(self, index):
        _, _, edge_attr = self.__get_input__(index)
        return edge_attr

    def __get_label__(self, index):
        id = format(index+self.start_index, self.num_digits)
        file_to_read = str(self.path)+'/snbs_'+str(id) + '.h5'
        hf = h5py.File(file_to_read, 'r')
        dataset_snbs = hf.get('snbs')
        snbs = np.array(dataset_snbs)
        hf.close()
        return torch.tensor(snbs).unsqueeze(1)

    def __getitem__(self, index):
        x, edge_index, edge_attr = self.__get_input__(index)
        y = self.__get_label__(index)
        data = gData(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        return data


class GCNNet01(torch.nn.Module):
    def __init__(self, num_classes=1, num_node_features=1):
        super(GCNNet01, self).__init__()
        self.conv1 = GCNConv(num_node_features, 4)
        self.conv2 = GCNConv(4, num_classes)
        self.endLinear = nn.Linear(num_classes, num_classes)
        self.endSigmoid = nn.Sigmoid()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        x = self.conv1(x=x, edge_index=edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x=x, edge_index=edge_index, edge_weight=edge_weight)
        x = self.endLinear(x)
        x = self.endSigmoid(x)
        return x


class GCNNet02(torch.nn.Module):
    def __init__(self, num_classes=1, num_node_features=1, num_nodes=10):
        super(GCNNet02, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, 4)
        self.conv3 = GCNConv(4, num_classes)
        self.endLinear = nn.Linear(num_classes, num_classes)
        self.endSigmoid = nn.Sigmoid()
        self.num_nodes = num_nodes
        self.num_classes = num_classes

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        x = self.conv1(x=x, edge_index=edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x=x, edge_index=edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.conv3(x=x, edge_index=edge_index, edge_weight=edge_weight)
        x = self.endLinear(x)
        x = self.endSigmoid(x)
        return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.endLinear.reset_parameters()


class GCNNet03(torch.nn.Module):
    def __init__(self, num_classes=1, num_node_features=1, num_nodes=10):
        super(GCNNet03, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv1_bn = nn.BatchNorm1d(16)
        self.conv2 = GCNConv(16, 4)
        self.conv2_bn = nn.BatchNorm1d(4)
        self.conv3 = GCNConv(4, num_classes)
        self.conv3_bn = nn.BatchNorm1d(num_classes)
        self.endLinear = nn.Linear(num_classes, num_classes)
        self.endSigmoid = nn.Sigmoid()
        self.num_nodes = num_nodes
        self.num_classes = num_classes

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        x = self.conv1(x=x, edge_index=edge_index, edge_weight=edge_weight)
        x = self.conv1_bn(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x=x, edge_index=edge_index, edge_weight=edge_weight)
        x = self.conv2_bn(x)
        x = F.relu(x)
        x = self.conv3(x=x, edge_index=edge_index, edge_weight=edge_weight) 
        x = self.endLinear(x)
        x = self.endSigmoid(x)
        return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.endLinear.reset_parameters()


class ArmaNet01(torch.nn.Module):
    def __init__(self, num_classes=1, num_node_features=1, num_nodes=10, num_layers=4, num_stacks=3):
        super(ArmaNet01, self).__init__()
        self.conv1 = ARMAConv(
            num_node_features, num_classes, num_stacks, num_layers)
        self.endLinear = nn.Linear(num_classes, num_classes)
        self.endSigmoid = nn.Sigmoid()
        self.num_nodes = num_nodes
        self.num_classes = num_classes

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        x = self.conv1(x=x, edge_index=edge_index,
                       edge_weight=edge_weight.float())
        x = self.endLinear(x)
        x = self.endSigmoid(x)
        return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.endLinear.reset_parameters()


class ArmaNet02(torch.nn.Module):
    def __init__(self, num_classes=1, num_node_features=1, num_nodes=10, num_layers=4, num_stacks=3):
        super(ArmaNet02, self).__init__()
        self.conv1 = ARMAConv(num_node_features, 16, num_stacks=num_stacks,
                              num_layers=num_layers, shared_weights=True, dropout=0.25)
        self.conv1_bn = nn.BatchNorm1d(16)
        self.conv2 = ARMAConv(16, num_classes, num_stacks=num_stacks,
                              num_layers=num_layers, shared_weights=True, dropout=0.25, act=None)
        self.conv2_bn = nn.BatchNorm1d(num_classes)
        self.endLinear = nn.Linear(num_classes, num_classes)
        self.endSigmoid = nn.Sigmoid()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x=x, edge_index=edge_index,
                       edge_weight=edge_weight.float())
        x = self.conv1_bn(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index=edge_index,
                       edge_weight=edge_weight.float())
        x = self.endLinear(x)
        x = self.endSigmoid(x)
        return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.endLinear.reset_parameters()


class SGNet01(torch.nn.Module):
    def __init__(self, num_classes=1, num_node_features=1, num_nodes=10, K=2):
        super(SGNet01, self).__init__()
        self.conv1 = SGConv(num_node_features, num_classes, K=K)
        self.K = K
        self.endLinear = nn.Linear(num_classes, num_classes)
        self.endSigmoid = nn.Sigmoid()
        self.num_nodes = num_nodes

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        x = self.conv1(x=x, edge_index=edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.endLinear(x)
        x = self.endSigmoid(x)
        return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.endLinear.reset_parameters()


class SGNet02(torch.nn.Module):
    def __init__(self, num_classes=1, num_node_features=1, num_nodes=10, K=2):
        super(SGNet02, self).__init__()
        self.conv1 = SGConv(num_node_features, 4, K=K)
        self.conv2 = SGConv(4, num_classes, K=K)
        self.endLinear = nn.Linear(num_classes, num_classes)
        self.endSigmoid = nn.Sigmoid()
        self.num_nodes = num_nodes

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        x = self.conv1(x=x, edge_index=edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x=x, edge_index=edge_index, edge_weight=edge_weight)

        x = self.endLinear(x)
        x = self.endSigmoid(x)
        return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.endLinear.reset_parameters()


class SGNet03(torch.nn.Module):
    def __init__(self, num_classes=1, num_node_features=1, num_nodes=10, K=2):
        super(SGNet03, self).__init__()
        self.conv1 = SGConv(num_node_features, 16, K=K)
        self.conv2 = SGConv(16, 4, K=K)
        self.conv3 = SGConv(4, num_classes, K=K)
        self.endLinear = nn.Linear(num_classes, num_classes)
        self.endSigmoid = nn.Sigmoid()
        self.num_nodes = num_nodes

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        x = self.conv1(x=x, edge_index=edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x=x, edge_index=edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.conv3(x=x, edge_index=edge_index, edge_weight=edge_weight)
        x = self.endLinear(x)
        x = self.endSigmoid(x)
        return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.endLinear.reset_parameters()


class TAGNet01(torch.nn.Module):
    def __init__(self, num_classes=1, num_node_features=1, num_nodes=10):
        super(TAGNet01, self).__init__()
        self.conv1 = TAGConv(num_node_features, 4)
        self.conv2 = TAGConv(4, num_classes)
        self.endLinear = nn.Linear(num_classes, num_classes)
        self.endSigmoid = nn.Sigmoid()
        self.num_nodes = num_nodes

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr.float()

        x = self.conv1(x=x, edge_index=edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x=x, edge_index=edge_index, edge_weight=edge_weight)
        x = self.endLinear(x)
        x = self.endSigmoid(x)
        return x


def initialize_model(model_name):
    if model_name == "GCNNet01":
        return GCNNet01()
    if model_name == "GCNNet02":
        return GCNNet02()
    if model_name == "GCNNet03":
        return GCNNet03()
    if model_name == "ArmaNet01":
        return ArmaNet01()
    if model_name == "ArmaNet02":
        return ArmaNet02()
    if model_name == "SGNet01":
        return SGNet01()
    if model_name == "SGNet02":
        return SGNet02()
    if model_name == "TAGNet01":
        return TAGNet01()
