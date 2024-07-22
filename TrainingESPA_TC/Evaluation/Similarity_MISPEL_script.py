'''
Created on Jul 5, 2022

@author: MAE82
'''
import pandas as pd
import os
import nibabel as nib
from skimage.metrics import structural_similarity as SSIM
from sklearn.metrics import mean_absolute_error as MAE
from scipy import stats
import numpy as np




def main(subject_names, image_file_name, Image_dir, step, save_number, mask, folder_name):
    
    SSIM_list = []
    SSIM_list_brain = []
    MAE_list = []
    MAE_within_brain_list = []
    Results = []
    
    for indj in range(0, len(Image_dir)):
        for indk in range(indj + 1, len(Image_dir)):
            SSIM_list = []
            MAE_list = []
            MAE_within_brain_list = []
            for indi in range(0, len(subject_names)):
                adr1 = os.path.join(save_number + '/Results', folder_name, Image_dir[indj], step, str(subject_names[indi]), image_file_name[indi])
                adr2 = os.path.join(save_number + '/Results', folder_name, Image_dir[indk], step, str(subject_names[indi]), image_file_name[indi])
        
                image_data1 = nib.load(adr1).get_fdata()
                image_data2 = nib.load(adr2).get_fdata()
        
                masked_image1 = image_data1[mask>0]
                masked_image2 = image_data2[mask>0]
        
                cleared_bckgrnd_image1 = np.multiply(image_data1, mask)
                cleared_bckgrnd_image2 = np.multiply(image_data2, mask)

                SSIM_list_brain.append(SSIM(cleared_bckgrnd_image1, cleared_bckgrnd_image2, multichannel = True))
                SSIM_list.append(SSIM(image_data1, image_data2, multichannel = True))
                MAE_list.append(MAE(image_data1.flatten(), image_data2.flatten()))
                MAE_within_brain_list.append(MAE(masked_image1.flatten(), masked_image2.flatten()))
            Results.append(pd.DataFrame(list(zip(subject_names, SSIM_list, SSIM_list_brain, MAE_list, MAE_within_brain_list)), columns = ["Subject_name", "SSIM",  "SSIM_brain", "MAE", "MAE_brain"]))
 
    return Results




def main_similarity(save_number, folder_name, subject_info):

    # read mask
    print("This one!")
    mask_path = "Dataset/cropped_JHU_MNI_SS_T1_Brain_Mask.nii.gz"
    mask = nib.load(mask_path).get_fdata()
    
    print("Start calculating similarities!")
    stream = ""
    
    data = pd.read_excel(subject_info)
    subject_names = data["SBJ_name"].tolist() 
    image_file_name = data["Nifti_fileName"].tolist()

    
    Image_dir = ["All_scanners_original_im0", "All_scanners_original_im1", "All_scanners_original_im2", "All_scanners_original_im3"]
                 
    step = "step2"
    Augmentation_similarity = main(subject_names, image_file_name, Image_dir, step, save_number, mask, folder_name)
    SSIM_original_mean = []
    SSIM_original_std = []
    
    SSIM_WB_original_mean = []
    SSIM_WB_original_std = []
    
    MAE_original_mean = []
    MAE_original_std = []
    
    MAE_WB_original_mean = []
    MAE_WB_original_std = []
    
    for i in range(len(Augmentation_similarity)):
        
        SSIM_original_mean.append(Augmentation_similarity[i]["SSIM"].mean())
        SSIM_original_std.append(Augmentation_similarity[i]["SSIM"].std())
        
        SSIM_WB_original_mean.append(Augmentation_similarity[i]["SSIM_brain"].mean())
        SSIM_WB_original_std.append(Augmentation_similarity[i]["SSIM_brain"].std())
    
        MAE_original_mean.append(Augmentation_similarity[i]["MAE"].mean())
        MAE_original_std.append(Augmentation_similarity[i]["MAE"].std())
    
        MAE_WB_original_mean.append(Augmentation_similarity[i]["MAE_brain"].mean())
        MAE_WB_original_std.append(Augmentation_similarity[i]["MAE_brain"].std())


    Image_dir = ["All_scanners_harmonized_im0", "All_scanners_harmonized_im1", "All_scanners_harmonized_im2", "All_scanners_harmonized_im3"]
                
    step = "step2"
    Harmonization_similarity = main(subject_names, image_file_name, Image_dir, step, save_number, mask, folder_name)
    
    SSIM_Harmonization_mean = []
    SSIM_Harmonization_std = []
    
    SSIM_WB_Harmonization_mean = []
    SSIM_WB_Harmonization_std = []
    
    MAE_Harmonization_mean = []
    MAE_Harmonization_std = []
    
    MAE_WB_Harmonization_mean = []
    MAE_WB_Harmonization_std = []
    
    for i in range(len(Harmonization_similarity)):
        
        SSIM_Harmonization_mean.append(Harmonization_similarity[i]["SSIM"].mean())
        SSIM_Harmonization_std.append(Harmonization_similarity[i]["SSIM"].std())
        
        SSIM_WB_Harmonization_mean.append(Harmonization_similarity[i]["SSIM_brain"].mean())
        SSIM_WB_Harmonization_std.append(Harmonization_similarity[i]["SSIM_brain"].std())
    
        MAE_Harmonization_mean.append(Harmonization_similarity[i]["MAE"].mean())
        MAE_Harmonization_std.append(Harmonization_similarity[i]["MAE"].std())
    
        MAE_WB_Harmonization_mean.append(Harmonization_similarity[i]["MAE_brain"].mean())
        MAE_WB_Harmonization_std.append(Harmonization_similarity[i]["MAE_brain"].std())

    
    # making dataframe 
    permutation = ["s1_s2", "s1_s3", "s1_s4", "s2_s3", "s2_s4", "s3_s4"]
    df_mean = pd.DataFrame(list(zip(permutation, SSIM_original_mean, SSIM_Harmonization_mean, SSIM_WB_original_mean, SSIM_WB_Harmonization_mean,
                                MAE_original_mean, MAE_Harmonization_mean,
                                MAE_WB_original_mean, MAE_WB_Harmonization_mean)),
               columns =["scanners", 'SSIM_RAW', 'SSIM_harm', 'SSIM_WB_RAW', 'SSIM_WB_harm', 'MAE_RAW', 'MAE_harm', 'MAE_WB_RAW', 'MAE_WB_harm'])

    df_mean.to_excel("./" + str(save_number) + "/Results/" + folder_name + "/sim_measure_mean.xlsx", index = False)
    
    df_std = pd.DataFrame(list(zip(permutation, SSIM_original_std, SSIM_Harmonization_std, SSIM_WB_original_std, SSIM_WB_Harmonization_std,
                                MAE_original_std, MAE_Harmonization_std,
                                MAE_WB_original_std, MAE_WB_Harmonization_std)),
               columns =["scanners", 'SSIM_RAW', 'SSIM_harm', 'SSIM_WB_RAW', 'SSIM_WB_harm', 'MAE_RAW', 'MAE_harm', 'MAE_WB_RAW', 'MAE_WB_harm'])
    
    df_std.to_excel("./" + str(save_number) + "/Results/" + folder_name + "/sim_measure_std.xlsx", index = False)
    
    #  paired t-test
    '''
    statistics, p_value = stats.ttest_rel(Augmentation_similarity["SSIM"], Harmonization_similarity["SSIM"])
    stream = stream +  str(statistics) + ", "+ str(p_value)
    #print(str(statistics) + ", "+ str(p_value))
    #print(stats.ttest_rel(Augmentation_similarity["SSIM"], Harmonization_similarity["SSIM"]))

    # Write the results
    path =  save_number + "/Results/" + step + "similarity.txt" 
    with open(path, 'w') as f:
        f.write(stream)
    '''
    print("Finished calculating similarities!")






