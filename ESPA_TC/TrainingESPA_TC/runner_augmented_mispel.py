'''
Created on Mar 13, 2022

@author: MAE82
'''
from Harmonization_method import Harmonization_method_main
import time
import os


# Setting GPUs
gpu_number = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number
Harmonization_method_main(gpu_number)



