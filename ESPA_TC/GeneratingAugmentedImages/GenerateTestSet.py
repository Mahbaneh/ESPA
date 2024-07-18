'''
Created on Aug 17, 2022

@author: MAE82
'''
import os
import pandas as pd
from GMM_augmentation_test_subtraction import GMM_Aug
import nibabel as nib
import numpy as np
import argparse



def generate_test_images_main(distribution_excel_adr, scanner_no): 
    
    if not os.path.isdir(dir_out):
        os.makedirs(dir_out)
        
    if not os.path.isdir(full_images_adr):
        os.makedirs(full_images_adr)
        
    if not os.path.isdir(nifti_test_adr):
        os.makedirs(nifti_test_adr)
        
    Augmentation = GMM_Aug(distribution_excel_adr, scanner_no)
    
    data = pd.read_excel(filename)
    mask = nib.load(mask_adr).get_fdata()
    Affine = np.eye(4)
    
    image_list = [[] for i in range(0, scanner_no)]
    slice_no = []
    scanner_no_list = []
    image_name = []

    counter = 0
    for ind in range(0, data.shape[0]):
        info = data.iloc[ind]
        image_adr = os.path.join(dir_in, info["Folder_name"], info["SBJ_name"], info["Nifti_fileName"])
        image = nib.load(image_adr).get_fdata()
        gmm_adr = os.path.join(dir_in, info["GGM_Folder_name"], info["SBJ_name"], info["GGM_File_name"] + ".sav")
        image_info = {"brain_image":image, "gmm_adr":gmm_adr, "mask":mask}                

        Augmented_images = []
        
        for indj in range(0, scanner_no):
            image_info["scanner_index"] = indj
            Aug_image = Augmentation(image_info)

            adr = os.path.join(full_images_adr, "image" + str(indj), info["SBJ_name"])
            if not os.path.isdir(adr):
                os.makedirs(adr)
            # write the full augmented image 
            nib.save(nib.Nifti1Image(Aug_image, Affine), os.path.join(adr, info["Nifti_fileName"]))
            Augmented_images.append(Aug_image)
            
        for indk in range(2, Aug_image.shape[2] - 2):  
            for indj in range(0, scanner_no):
                Aug_image = Augmented_images[indj]
                slice = np.expand_dims(Aug_image[:, : , indk], axis=0)
                adr = os.path.join(nifti_test_adr, "image_info_" + str(counter) + "_image" + str(indj + 1) + ".nii.gz")
                nib.save(nib.Nifti1Image(slice, Affine), adr)
                image_list[indj].append(adr)
            slice_no.append(indk)
            scanner_no_list.append(info["ScannerID"])
            image_name.append(info["Nifti_fileName"])
            counter +=1
            
    total_list = [image_name, scanner_no_list, slice_no]
    col_names = ["image_name", "scanner", "slice"]
    
    for indi in range(0, len(image_list)):
        total_list.append(image_list[indi])
        col_names.append("nifi_image" + str(indi + 1))
    
    all_data = np.array(total_list)
    all_data = all_data.transpose()
    
    df = pd.DataFrame(all_data, columns = col_names)
    df.to_excel(os.path.join(dir_out, "selected_images_all_epoch_nifti_mispel_augmented.xlsx"))

    print("Finished!")
    
    
def parse_option():
    
    
    parser = argparse.ArgumentParser('argument for training')
    
    parser.add_argument('--filename', type=str, default="./Dataset/Images_test.xlsx",
                        help='Adress to list of input images.')
    
    parser.add_argument('--dir_in', type=str, default = "./Dataset/FinalFiles",
                        help='Directory of data.')

    parser.add_argument('--dir_out', type=str, default="./Dataset/Data_For_Loader_test/saved_data",
                        help='Directory to output augmented images.')
    
    parser.add_argument('--mask_adr', type=str, default="./Dataset/cropped_JHU_MNI_SS_T1_Brain_Mask.nii.gz",
                        help='adress to the brain mask.')
    
    parser.add_argument('--distribution_excel_adr', type=str, default="./Dataset/Distributions/CV_0",
                        help='Directory to the distribution of differences for tissue-type parameters.')
 
    
    opt = parser.parse_args()

    return opt
    
    

opt = parse_option()
#print(opt.filename)
filename = opt.filename
dir_in = opt.dir_in
dir_out = opt.dir_out
mask_adr = opt.mask_adr

full_images_adr = os.path.join(dir_out, "full_images")
nifti_test_adr = os.path.join(dir_out, "test_nifti")
distribution_excel_adr = opt.distribution_excel_adr

scanner_no = 4
generate_test_images_main(distribution_excel_adr, scanner_no)
    


