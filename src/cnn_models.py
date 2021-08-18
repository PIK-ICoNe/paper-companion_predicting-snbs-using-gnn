import os
import h5py
# import pandas as pd
import numpy as np


import torch as torch
# import torch.nn as nn


from torch.utils.data.dataset import Dataset

# import torchvision
# import torchvision.models as models


# TO DELTE ###################
#from myResnet import resnet18
#############################
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


class cnn_snbs(Dataset):
    def __init__(self, grid_path, slice_index = slice(0,0) , num_input_features=1):
        if slice_index.stop == 0:
            self.data_len, self.start_index, digits = get_length_of_dataset(grid_path)
        else:
            _, _, digits = get_length_of_dataset(grid_path)
            self.start_index = slice_index.start + 1
            self.data_len = slice_index.stop - slice_index.start + 1

        self.path = grid_path
        self.num_digits = '0' + str(digits.__str__().__len__())
        self.num_classes = 1
        self.num_input_features = num_input_features

    def __len__(self):
        return self.data_len

    def num_classes(self):
        return self.num_classes
    
    def __get_input__(self, index):
        if index +1 > self.data_len:
            print('Error in dataset: Trying to access invalid element')
        else:
            id = format(index+self.start_index, self.num_digits)
            file_to_read = str(self.path)+'/grid_cnn_data_'+str(id) + '.h5'
            hf = h5py.File(file_to_read, 'r')
            # read in sources/sinks
            dataset_P = hf.get('P')
            P = torch.tensor(np.array(dataset_P))
            # read in L
            dataset_L = hf.get('L')
            L = torch.tensor(np.array(dataset_L))
            hf.close()
            if self.num_input_features == 1:
                return torch.cat((L,P.unsqueeze(0)),dim=0).unsqueeze(0)
            if self.num_input_features == 2:
                return torch.stack((L,torch.diag(P)),dim=0)

    def __label__(self, index):
        id = format(index+self.start_index, self.num_digits)
        file_to_read = str(self.path)+'/snbs_'+str(id) + '.h5'
        hf = h5py.File(file_to_read, 'r')
        dataset_snbs = hf.get('snbs')
        snbs = np.array(dataset_snbs)
        hf.close()
        return torch.tensor(snbs)

    def __getitem__(self, index):
        return (self.__get_input__(index), self.__label__(index))