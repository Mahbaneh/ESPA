'''
Created on Mar 15, 2022

@author: MAE82
'''

import torch
from torch.utils import data
import pickle
import time
import nibabel as nib
import os


class HarmonizationDataSet(data.Dataset):
    
    def __init__(self,
                 mask,
                 adr_data,
                 scanner_no
                 ):
        
        self.adr_data_names = []
        self.scanner_no = scanner_no

        for i in range(1, scanner_no + 1):
            self.adr_data_names.append("nifi_image" + str(i))
        self.adrs = adr_data[self.adr_data_names].to_numpy()


    def __len__(self):
        return (int)(len(self.adrs))
    

    def __getitem__(self, index: int):

        images = []
        
        for i in range(0, self.scanner_no):
            image = nib.load(self.adrs[index, i]).get_fdata()
            images.append(torch.tensor(image, dtype = torch.float))
 
        return images, index


'''
class HarmonizationDataSet(data.Dataset):
    
    def __init__(self,
                 mask,
                 adr_data
                 ):

        self.image1_adr = list(adr_data["nifi_image1"])
        self.image2_adr = list(adr_data["nifi_image2"])
        self.labels = list(adr_data["scanner"])

    def __len__(self):
        return len(self.image1_adr)

    def __getitem__(self, index: int):

        # Load nifti
        time10 = time.time()
        image1 = nib.load(self.image1_adr[index]).get_fdata()
        image1 = torch.tensor(image1, dtype = torch.float)
        #print(image1.shape)
        
        image2 = nib.load(self.image2_adr[index]).get_fdata()
        image2 = torch.tensor(image2, dtype = torch.float)
        
        label = self.labels[index]
        time20 = time.time()
        #print("Pickle:" + str(time20 - time10))
        return [image1, image2, image1, image2]# ***Mah

'''

'''
class HarmonizationDataSet(data.Dataset):
    
    def __init__(self,
                 mask,
                 adr_data
                 ):

        self.pickle_adr = list(adr_data["pickle_adr"])

    def __len__(self):
        return len(self.pickle_adr)

    def __getitem__(self, index: int):

        # Load pickle
        time10 = time.time()
        with open(self.pickle_adr[index], 'rb') as handle:
            Data = pickle.load(handle)
            handle.close()

        time20 = time.time()
        print("Pickle:" + str(time20 - time10))
        
        A = nib.load("./slice100_5d27474ff8_UCDavis.nii.gz").get_data()
        time30 = time.time()
        print("nibabel:" + str(time30 - time20))
        return [Data[0], Data[2]], Data[1]
'''        


