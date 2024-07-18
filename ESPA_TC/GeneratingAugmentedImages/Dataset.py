'''
Created on Mar 15, 2022

@author: MAE82
'''
import os 
import torch
from torch.utils import data
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
seed = 1024

def set_seeds():
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def set_GPUs():
    # Setting GPUs
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    
set_seeds()
set_GPUs()


class HarmonizationDataSet(data.Dataset):
    
    def __init__(self,
                 mask,
                 adr_data,
                 transform,
                 ):
        
        self.image_adresses = list(adr_data["adr"])
        self.slice_scanner_ID = list(adr_data["scanner"])
        self.slice_no = list(adr_data["slice"])
        self.transform = transform
        self.mask = mask
        self.gmm_addresses = list(adr_data["gmm"])

    def __len__(self):
        return len(self.slice_scanner_ID)

    def __getitem__(self, index: int):

        image_adr = self.image_adresses[index]
        label = self.slice_scanner_ID[index]
        gmm_adr = self.gmm_addresses[index]
        slice_index = self.slice_no[index]

        # Load input image
        image_brain_data = nib.load(image_adr).get_fdata()
        image_slice = image_brain_data[:, :, self.slice_no[index]]

        # Augmenttations
        if self.transform is not None:
            final_image = self.transform(image_brain_data, self.mask,
                                          slice_index, 
                                          gmm_adr,
                                          image_slice)
        
        # Add dimension and convert to tensor  
        for ind in range(0, len(final_image)):  
            final_image[ind] = np.expand_dims(final_image[ind], axis=0)
            final_image[ind] = torch.from_numpy(final_image[ind])
            final_image[ind] = final_image[ind].float()
        
        return final_image, index


