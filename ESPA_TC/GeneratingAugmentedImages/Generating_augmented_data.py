from __future__ import print_function

import os
import sys
import argparse

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
from Dataset import HarmonizationDataSet

from util import CropTransform
from GMM_augmentation_subtraction import GMM_Aug
import random
import numpy as np
import pandas as pd
import nibabel as nib


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
def set_seeds_torch(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_GPUs():
    # Setting GPUs
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def parse_option():
    
    
    parser = argparse.ArgumentParser('argument for training')
    
    parser.add_argument('--scanner_no', type=int, default=4,
                        help='The Number of scanners.')
    
    parser.add_argument('--mask_adr', type=str, default = "Dataset/cropped_JHU_MNI_SS_T1_Brain_Mask.nii.gz",
                        help='The address to mask file.')

    parser.add_argument('--no_axial_slices', type=int, default=152,
                        help='Number of axial slices in images in their template space.')
    
    parser.add_argument('--dist_adr', type=str, default="./Dataset/Distributions/CV_0/",
                        help='The directory to distributions excel file.')
    
    parser.add_argument('--train_image_input_info', type=str, default="./Dataset/Images_train.xlsx",
                        help='The excel file for images of train set.')
    
    parser.add_argument('--validation_image_input_info', type=str, default="./Dataset/Images_validation.xlsx",
                        help='The excel file for images of validation set.')   
                        
    parser.add_argument('--folder_path', type=str, default="./Dataset/Data_For_Loader_trainvalidation",
                        help='Path for saving Augmeneted train and validation data.')

    parser.add_argument('--batch_size', type=int, default= 8,
                        help='batch_size.')
    
    parser.add_argument('--num_workers', type=int, default= 0,
                        help='Number of workers to use.')
    
    opt = parser.parse_args()

    return opt



def genearting_data_validation(loader, opt, counter):

    Affine = np.eye(4)
    image_list_all = [[] for indi in range(0, opt.scanner_no)]
    
    with torch.no_grad():
        for _, (images, _) in enumerate(loader):
            image_no = images[0].shape[0]
            for indi in range(0, image_no):
                for indj in range(0, opt.scanner_no):
            
                    path = os.path.join(opt.folder_path, "saved_data/validation_nifti" + str(opt.seed))
                    if (os.path.isdir(path) == False):
                        os.mkdir(os.path.join(opt.folder_path, "saved_data/validation_nifti" + str(opt.seed)))
                        
                    path = os.path.join(path, "image_info_" + str(counter) + "_image" + str(indj + 1) + ".nii.gz")
                    image = images[indj][indi].to("cpu")
                    image = image.numpy()
                    nib.save(nib.Nifti1Image(image, Affine),  path)
                    image_list_all[indj].append(path)
              
                counter += 1  
    return counter, image_list_all


def genearting_data_train(loader, opt, counter):
    
    Affine = np.eye(4)
    image_list_all = [[] for indi in range(0, opt.scanner_no)]
    
    for _, (images, _) in enumerate(loader):
        image_no = images[0].shape[0]
        for indi in range(0, image_no):
            for indj in range(0, opt.scanner_no):
                path = os.path.join(opt.folder_path, "saved_data/train_nifti" + str(opt.seed))
                if (os.path.isdir(path) == False):
                    os.mkdir(os.path.join(opt.folder_path, "saved_data/train_nifti" + str(opt.seed)))
                    
                path = os.path.join(path + "/image_info_" + str(counter) + "_image" + str(indj + 1) + ".nii.gz")
                image = images[indj][indi].to("cpu")
                image = image.numpy()
                nib.save(nib.Nifti1Image(image, Affine),  path)
                image_list_all[indj].append(path)
            counter += 1
              
    return counter, image_list_all


def Read_data_info(opt, excel_adr):
    
    image_info = pd.read_excel(excel_adr)
    
    # Generate the addresses 
    image_adr_list = []
    gmm_adr_list = []
    adr_scanner_id = []
    adr_slice_id = []
    
    for indi in range(0, image_info.shape[0]):
        
        # Prepare address of the slice
        image_adr = os.path.join("Dataset/FinalFiles", image_info["Folder_name"][indi]
                                     , image_info["SBJ_name"][indi], image_info["Nifti_fileName"][indi])
        image_adr_list.extend([image_adr] * opt.no_axial_slices) 
        
        # Prepare address of the GMM
        gmm_adr = os.path.join("Dataset/FinalFiles", image_info["GGM_Folder_name"][indi]
                                     , image_info["SBJ_name"][indi], image_info["GGM_File_name"][indi] + ".sav")
        gmm_adr_list.extend([gmm_adr] * opt.no_axial_slices) 
        
        adr_scanner_id.extend([image_info["ScannerID"][indi]] * opt.no_axial_slices)
        adr_slice_id.extend(list(range(2, opt.no_axial_slices + 2)))
        
    all_list = list(zip(image_adr_list, adr_scanner_id, adr_slice_id, gmm_adr_list))
        
    return all_list


def set_loader(opt, train_adr_data, validation_adr_data, mask):
    
    # Defining the augmentations. 
    train_transform = transforms.Compose([GMM_Aug(opt.dist_adr, opt.scanner_no)])

    # Loading the dataset.  
    train_dataset = HarmonizationDataSet(mask, train_adr_data, transform=CropTransform(train_transform, opt.scanner_no))
    validation_dataset = HarmonizationDataSet(mask, validation_adr_data,transform=CropTransform(train_transform, opt.scanner_no))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle = False,
        num_workers=opt.num_workers, pin_memory=True, sampler = None)
    
    valid_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=opt.batch_size, shuffle = False,
        num_workers=opt.num_workers, pin_memory=True, sampler = None)
    
    return train_loader, valid_loader
    

def read_mask(opt):
    
    mask = nib.load(opt.mask_adr).get_fdata()
    return mask


def Shuffle_data(all_adrs):
    
    # Shuffle indices
    mylist = list(range(0, len(all_adrs)))
    random.shuffle(mylist)
    all_data = [all_adrs[i] for i in mylist]
    
    return all_data


def generatingData_train_validation(opt, train_loader, valid_loader, counter_train, counter_validation):
        
    # Make excel file of slices for each generated dataset.
    train_filename = os.path.join(opt.folder_path, "saved_data/training_images_one_epoch_nifti_" + str(opt.seed) + ".xlsx")
    validation_filename = os.path.join(opt.folder_path, "saved_data/validation_images_one_epoch_nifti_" + str(opt.seed) + ".xlsx")
    image_info = pd.read_excel(os.path.join(opt.folder_path, "saved_data/training_images_one_epoch.xlsx"))

    # Generate augmented data for train
    set_seeds(opt.seed)
    counter_train, lists= genearting_data_train(train_loader, opt, counter_train)
        
    for indi in range(0, opt.scanner_no):
        image_info["nifi_image" + str(indi + 1)] = lists[indi]
            
    image_info.to_excel(train_filename)
        
    # Generate augmented data for validation
    if opt.seed == 0:
        image_info = pd.read_excel(os.path.join(opt.folder_path, "saved_data/validation_images_one_epoch.xlsx"))
        counter_validation, lists = genearting_data_validation(valid_loader, opt, counter_validation)
        for indi in range(0, opt.scanner_no):
            image_info["nifi_image" + str(indi + 1)] = lists[indi]
        image_info.to_excel(validation_filename)
        
    

def Generating_augmented_data_main(seed):

    step1_counter_train = 0
    step1_counter_validation = 0
    
    opt = parse_option()
    opt.seed = seed
    set_seeds(opt.seed)
    set_GPUs()
    
    # Read mask
    mask = read_mask(opt)

    # Read train data
    all_adrs = Read_data_info(opt, opt.train_image_input_info)
    
    # Shuffle data for once. 
    adr_list_train = Shuffle_data(all_adrs)
    
    # Read validation data
    all_adrs = Read_data_info(opt, opt.validation_image_input_info)
    
    # Shuffle data for once.
    adr_list_validation = Shuffle_data(all_adrs)
    
    # Convert to pandas
    train_adr_data = pd.DataFrame(adr_list_train, columns = ['adr', 'scanner', 'slice', 'gmm'])
    validation_adr_data = pd.DataFrame(adr_list_validation, columns = ['adr', 'scanner', 'slice', 'gmm'])
    
    # Write down the excel files train and validation slices. 
    dir = os.path.join(opt.folder_path, "saved_data")
    if os.path.exists(dir) == False:
        os.mkdir(dir)
    train_adr_data.to_excel(os.path.join(dir, "training_images_one_epoch.xlsx"))
    validation_adr_data.to_excel(os.path.join(dir, "validation_images_one_epoch.xlsx"))
   
    # build data loader
    train_loader, valid_loader = set_loader(opt, train_adr_data, validation_adr_data, mask)
    
    # setting seeds
    set_seeds_torch(opt.seed)
   
    # generating data
    generatingData_train_validation(opt, train_loader, valid_loader, step1_counter_train, step1_counter_validation)
 
