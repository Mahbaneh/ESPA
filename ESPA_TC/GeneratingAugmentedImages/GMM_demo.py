

import nibabel as nib
import os
import pandas as pd
from sklearn import mixture
import numpy as np
import pickle
n_components = 3
gmm_seed = 1234

def set_GPUs():
    # Setting GPUs
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def create_dir(mypath):
    """Create a directory if it does not exist."""
    try:
        os.makedirs(mypath)
    except OSError as exc:
        if os.path.isdir(mypath):
            pass
        else:
            raise

def fit_gmm(x, n_components=3, gmm_seed = 1234):
    """ Fit the GMM to the data
    :param x:  non-zero values of image. Should be of shape (N,1). Make sure to use X=np.expand_dims(data(data>0),1)
    :param n_components: number of components in the mixture. Set to None to select the optimal component number based
           on the BIC criterion. Default: 3
    :return: GMM model, fit to the data
    """
    
    gmm = mixture.GaussianMixture(n_components=n_components, covariance_type='diag', tol=1e-3, random_state= gmm_seed)

    return gmm.fit(x)


def GMM_main(excel_adr, mask_adr):
    
    # 1. Read excel file
    excel_data = pd.read_excel(excel_adr)
    
    # 2. Read mask
    mask = nib.load(mask_adr).get_fdata() 
    
    # 3. make adress of images
    image_adrs = []
    for indi in range(0, excel_data.shape[0]):
        image_adrs.append(os.path.join("Dataset/FinalFiles", excel_data["Folder_name"][indi]
                                     , excel_data["SBJ_name"][indi], excel_data["Nifti_fileName"][indi])) 

    # 4. Get the GMM of images. 
    for indi in range(0, len(image_adrs)):
        
        adr = image_adrs[indi]
        image_data = nib.load(adr).get_fdata()
        
        # 4.1. Mask the image
        masked_image = image_data * mask
        data = masked_image[mask > 0]
        flattened_data = np.expand_dims(data, 1)
        
        # 4.2. Get the gmm
        gmm = fit_gmm(flattened_data, n_components, gmm_seed)
        
        # 4.3. Make the main directory 
        create_dir(excel_data["GGM_Folder_name"][indi])
        create_dir(os.path.join("Dataset/FinalFiles", excel_data["GGM_Folder_name"][indi]
                                     , excel_data["SBJ_name"][indi]))
        
        # make the address 
        gmm_adr = os.path.join("Dataset/FinalFiles", excel_data["GGM_Folder_name"][indi]
                                     , excel_data["SBJ_name"][indi], excel_data["GGM_File_name"][indi] + ".sav")

        #Saving the GMM
        pickle.dump(gmm, open(gmm_adr, 'wb'))
        
        




