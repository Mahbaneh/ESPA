'''
Created on Mar 15, 2022

@author: MAE82
'''

import torch
from torch.utils import data

class ScannerDataset(data.Dataset):
    def __init__(self,
                 images, # numpy array of stacked axial slices
                ):
        self.dataset = images

    def __len__(self):
        return self.dataset.shape[2]

    def __getitem__(self, index: int):
        
        axial_slice = torch.tensor(self.dataset[:, :, index], dtype = torch.float)
        return axial_slice[None, :, :]
    
    
class ScannerDataset_withsliceIndex(data.Dataset):
    def __init__(self,
                 images, # numpy array of stacked axial slices
                 indices # index of axial slice
                ):
        self.dataset = images
        self.indices = indices

    def __len__(self):
        return self.dataset.shape[2]

    def __getitem__(self, index: int):
        
        axial_slice = torch.tensor(self.dataset[:, :, index], dtype = torch.float)
        return [axial_slice[None, :, :], self.indices[index]]

