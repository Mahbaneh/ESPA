from __future__ import print_function

import os
import math
import torch
import torch.optim as optim
import random
import numpy as np
import openpyxl
seed = 1024


def set_seeds():
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_GPUs():
    # Setting GPUs
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

set_seeds()
set_GPUs()



class CropTransform:
    
    def __init__(self, transform, number_of_transforms):
        self.number_of_transforms = number_of_transforms
        self.transform = transform
        

    def __call__(self, brain_image, mask, slice_no, gmm_adr, image_slice):

        
        image_list = []
        arg_dict = {"brain_image": brain_image,
                     "mask": mask,
                     "slice_no": slice_no, "gmm_adr": gmm_adr}

        for indi in range(0, self.number_of_transforms):
            arg_dict["scanner_index"] = indi
            image_list.append(self.transform(arg_dict))

        return image_list
    

class FourCropTransform:
    
    def __init__(self, transform):
        self.transform = transform
        

    def __call__(self, brain_image, mask, slice_no, gmm_adr, image_slice):

    
        arg_dict = {"brain_image": brain_image,
                     "mask": mask,
                     "slice_no": slice_no, "gmm_adr": gmm_adr}
        
        transformed_image1 = self.transform(arg_dict)
        transformed_image2 = self.transform(arg_dict)
        transformed_image3 = self.transform(arg_dict)
        
        return [image_slice, transformed_image1, transformed_image2, transformed_image3]



