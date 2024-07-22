'''
Created on Jun 27, 2022

@author: MAE82
'''
from Harmonization_method import parse_option, Read_data_info, read_mask, set_loader_MISPEL, set_model
import torch
import pandas as pd
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from collections import OrderedDict


seed = 1024
def set_seeds():

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seeds()



def create_dir(mypath):
    """Create a directory if it does not exist."""
    try:
        os.makedirs(mypath)
    except OSError as exc:
        if os.path.isdir(mypath):
            pass
        else:
            raise


def synthesize_image(model, loader, scanners):
    
    model.eval()
    harmonized_images = [[] for i in range(len(scanners))]
    original_images = [[] for i in range(len(scanners))]
    with torch.no_grad():
        for idx, (images, index) in enumerate(loader):
            if torch.cuda.is_available():
                for i in range(0, len(images)):
                    images[i] = images[i].cuda(non_blocking=True)
                
            _, reconstructed_images = model(images, 1)   
            for i in range(0, len(reconstructed_images)):
                images[i] = images[i].to("cpu")
                images[i] = images[i].tolist()
                for j in range(0, len(images[i])):
                    images[i][j] = np.array(images[i][j])
                    original_images[i].append(images[i][j])
                    
                reconstructed_images[i] = reconstructed_images[i].to("cpu")
                reconstructed_images[i] = reconstructed_images[i].tolist()
                for j in range(0, len(reconstructed_images[i])):
                    reconstructed_images[i][j] = np.array(reconstructed_images[i][j])
                    harmonized_images[i].append(reconstructed_images[i][j])

    return original_images, harmonized_images



def writing_images(original_images, harmonized_images, slice_number, address, step, save_number, scanners, folder_name):
    
    path_orig = [save_number + "/Results/" + folder_name + "/All_scanners_original_im" + str(i) + "/" + step for i in range(len(scanners))]
    path_harm = [save_number + "/Results/" + folder_name + "/All_scanners_harmonized_im" + str(i) + "/" + step for i in range(len(scanners))]
    
    for indi in range(0, len(path_orig)):
        create_dir(path_orig[indi])
        create_dir(path_harm[indi])
        
    img = nib.load("./Dataset/cropped_JHU_MNI_SS_T1_Brain_Mask.nii.gz")
    Affine = img.affine
    imag_data = img.get_fdata()
    
    adrs = pd.read_excel(address)
    
    # Folder_name
    for indi in range(0, adrs.shape[0]):
        
        # read images
        info = adrs.iloc[[indi]]
        scanner_slices = [[] for i in range(len(scanners))]
        harm_slices = [[] for i in range(len(scanners))]
        
        scanner_images = [np.zeros((imag_data.shape)) for i in range(len(scanners))]
        harm_images = [np.zeros((imag_data.shape)) for i in range(len(scanners))]
        
        for indj in range(len(scanners)):
            scanner_slices[indj] = original_images[indj][indi * slice_number : (indi + 1) * slice_number]
            harm_slices[indj] = harmonized_images[indj][indi * slice_number : (indi + 1) * slice_number]
            
            scanner_slices[indj] = np.squeeze(np.array(scanner_slices[indj]))
            harm_slices[indj] = np.array(harm_slices[indj])
            
            # padd images
            scanner_slices[indj] = np.swapaxes(scanner_slices[indj],0,2)
            scanner_slices[indj] = np.swapaxes(scanner_slices[indj],0,1)
            scanner_images[indj][:, :, 2:slice_number + 2] = scanner_slices[indj]
            
            harm_slices[indj] = np.swapaxes(harm_slices[indj],0,2)
            harm_slices[indj] = np.swapaxes(harm_slices[indj],0,1)
            harm_images[indj][:, :, 2:slice_number + 2] = harm_slices[indj]
            
            scanner_images[indj] = nib.Nifti1Image(scanner_images[indj], Affine)
            harm_images[indj] = nib.Nifti1Image(harm_images[indj], Affine)
            
            create_dir(os.path.join(path_orig[indj], str(info["SBJ_name"].values[0])))
            create_dir(os.path.join(path_harm[indj], str(info["SBJ_name"].values[0])))
            
            nib.save(scanner_images[indj], os.path.join(path_orig[indj], str(info["SBJ_name"].values[0]), info["Nifti_fileName"].values[0]))
            nib.save(harm_images[indj], os.path.join(path_harm[indj], str(info["SBJ_name"].values[0]), info["Nifti_fileName"].values[0]))



def load_model(model, model_adr, method_type, opt):

    if method_type == 1:
        # This is from augmented_MISPEL
        checkpoint = torch.load(model_adr)
        model.load_state_dict(checkpoint['model'])
        
    elif method_type == 2:
        # This is from augmented_2image_model
        checkpoint = torch.load(model_adr)
        saved_model_stats = checkpoint['model']
        model = customized_load_stat_dict_2_Augmented_images(model, saved_model_stats, opt)

    return model


def main_harmonized_images(excel_adr, mask_adr, file_name, model_adr, worker_no, step, save_number, gpu_number, scanners, address, method_type, folder_name, no_slices):
 
    opt = parse_option(gpu_number)
    
    # Read mask
    mask = read_mask(opt)

    # Write the info of the slices as excel file
    train_adr_data = pd.read_excel(excel_adr)
    
    # Build data loader: I want the loader to do the transformation
    # Shuffle in dataloader MUST be FALSE
    loader = set_loader_MISPEL(opt, train_adr_data, mask)

    # build model and criterion
    model = set_model(opt)
    
    # load model
    load_model(model, model_adr, method_type, opt)
    
    # harmonizing
    original_images, harmonized_images = synthesize_image(model, loader, scanners)
   
    # writing images
    writing_images(original_images, harmonized_images, no_slices, address, step, save_number, scanners, folder_name)

    
    
def get_subject_name(x):
    return x.split("/")[-2]



def HarmonizingImages_main(save_number, model_adr, gpu_number, method_type, address, mask_adr, excel_adr, folder_name, no_slices, image_list):
 
    slice_list = [90]   
    worker_no = 0 
    scanners = ["ge", "philips", "prisma", "trio"] 
    file_name = "paired_" + scanners[0] +  "_" +  scanners[1]

    step = "step2"
    print("Started harmonizing images!")
    main_harmonized_images(excel_adr, mask_adr, file_name, model_adr, worker_no, step, save_number, gpu_number, scanners, address, method_type, folder_name, no_slices)
    print("Finished harmonizing images!")

    


    

