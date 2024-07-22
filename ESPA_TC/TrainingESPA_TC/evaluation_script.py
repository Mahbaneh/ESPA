'''
Created on Mar 13, 2022

@author: MAE82
'''

import os
from HarmonizingImages_script import HarmonizingImages_main
from Evaluation.Similarity_script import main_similarity
from Evaluation.plotting_losses_script import main_plotting_losses



def evaluation_main(gpu_number, model_name):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number
    save_number = "save" + gpu_number
    
    # call for harmonizing images
    HarmonizingImages_main(save_number, model_name, gpu_number)

    # call for similarity evaluation
    main_similarity(save_number)
    
    # call for plotting images
    main_plotting_losses(save_number, model_name)
    
    

  
gpu_number = "0"    
model_name = "SimCLR_cifar10_UNET_lr_0.01_decay_0.0001_bsz_64_temp_0.5_trial_0_cosine"
evaluation_main(gpu_number, model_name)


'''
gpu_number = "1"    
model_name = "SimCLR_cifar10_UNET_lr_0.001_decay_0.0001_bsz_64_temp_0.5_trial_0_cosine"
evaluation_main(gpu_number, model_name)


gpu_number = "2"    
model_name = "SimCLR_cifar10_UNET_lr_0.0005_decay_0.0001_bsz_64_temp_0.5_trial_0_cosine"
evaluation_main(gpu_number, model_name)



gpu_number = "3"    
model_name = "SimCLR_cifar10_UNET_lr_1e-05_decay_0.0001_bsz_64_temp_0.5_trial_0_cosine"
evaluation_main(gpu_number, model_name)
'''