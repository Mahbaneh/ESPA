'''
Created on Mar 13, 2022

@author: MAE82
'''

import os
from HarmonizingImages import HarmonizingImages_main
from Evaluation.Similarity_MISPEL_script import main_similarity
from Evaluation.test_preparing_files import validation_preparing_files_main


def evaluation_main(gpu_number, model_name, method_type, no_slices):

    save_number = "save" + gpu_number
    excel_adr = "./Dataset/selected_images_all_epoch_nifti_paired.xlsx"
    mask_adr = "./Dataset/cropped_JHU_MNI_SS_T1_Brain_Mask.nii.gz"
    address =  "Dataset/one_selected_scanner_data_paired.xlsx"
    folder_name = "Paired"
    
    # call for harmonizing images
    image_list = ["f4a3846a7a_UCDavis.nii.gz"]
    HarmonizingImages_main(save_number, model_name, gpu_number, method_type, address, mask_adr, excel_adr, folder_name, no_slices, image_list)

    # call for similarity evaluation
    subject_info_adr = "Dataset/one_selected_scanner_data_paired.xlsx"
    main_similarity(save_number, folder_name, subject_info_adr)

    
      
def evaluation_main_validation(gpu_number, model_name, method_type, no_slices, CV_no):

    validation_preparing_files_main(CV_no)
    save_number = "save" + gpu_number
    excel_adr = "./Dataset/validation_images_one_epoch_nifti_" + CV_no + "_sorted.xlsx"
    mask_adr = "./Dataset/cropped_JHU_MNI_SS_T1_Brain_Mask.nii.gz"
    address = "./Dataset/one_selected_scanner4_data_validation.xlsx"
    folder_name = "Validation"
    
    # call for harmonizing images
    image_list = ["sub-OAS30674_ses-d1011_run-01_T1w.nii.gz"]
    HarmonizingImages_main(save_number, model_name, gpu_number, method_type, address, mask_adr, excel_adr, folder_name, no_slices, image_list)

    # call for similarity evaluation
    subject_info_adr = "./Dataset/one_selected_scanner4_data_validation.xlsx"
    main_similarity(save_number, folder_name, subject_info_adr)
    
    
        
def evaluation_main_test(gpu_number, model_name, method_type, no_slices, CV_no ):

    save_number = "save" + gpu_number
    excel_adr = "./Dataset/test_selected_images_all_epoch_nifti_mispel_augmented_" + CV_no + ".xlsx"
    mask_adr = "./Dataset/cropped_JHU_MNI_SS_T1_Brain_Mask.nii.gz"
    address = "./Dataset/one_selected_scanner4_data_test.xlsx"
    folder_name = "Test"
    
    # call for harmonizing images
    image_list = ["sub-OAS30881_ses-d0304_run-01_T1w.nii.gz"]
    HarmonizingImages_main(save_number, model_name, gpu_number, method_type, address, mask_adr, excel_adr, folder_name, no_slices, image_list)

    # call for similarity evaluation
    subject_info_adr = "Dataset/one_selected_scanner4_data_test.xlsx"
    main_similarity(save_number, folder_name, subject_info_adr)
    

CV_no = "CV0"
gpu_number = "0"  
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number
method_type =  1
no_slices = 152
model_name = "/projects/Mahbaneh/SPA/9_Augmented_MISPEL_Cleaned/src/save" + str(gpu_number) + "/SupCon/cifar10_models/model/training_step2/ckpt_epoch_600.pth"
#evaluation_main(gpu_number, model_name, method_type, no_slices)
evaluation_main_test(gpu_number, model_name, method_type, no_slices, CV_no)
#evaluation_main_validation(gpu_number, model_name, method_type, no_slices, CV_no)

