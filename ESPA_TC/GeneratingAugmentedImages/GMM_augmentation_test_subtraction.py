
"""
The original code in this file was from https://github.com/icometrix/gmm-augmentation.
We made some changes to use it for our project.
"""


from __future__ import print_function
import os
import torch
import random
import numpy as np
import pickle
import pandas as pd
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

class GMM_Aug():
    
    def __init__(self, dist_adr, scanner_no):
        ########### constants ########### 
        self.components = 3
        self.augmentation_randomness_seed = 1024
        
        ########### Variables ###########
        self.brain_image = np.array([])
        self.mask = np.array([])
        self.slice_scanner_ID = 0
        self.slice_no = 0
        self.gmm_adr = []
        
        # read distributions
        self.dist_adr = dist_adr
        self.scanner_no = scanner_no
        self.distributions = []
        self.read_distributions()
        
        
    def add_to_dic(self, label, df, dict_parameters):
        
        values = df[df["dist_type"] == label].values.tolist()[0][1:]
        values = [float(val) for val in values]
        dict_parameters[label] = values 
        return dict_parameters
        
        
    def read_distributions(self):
        
        for indi in range(0, self.scanner_no):
            df = pd.read_excel(os.path.join(self.dist_adr, "scanner" + str(indi) + ".xlsx"))
            dict_parameters = {}
            
            dict_parameters = self.add_to_dic("uniform_dis_mean_start", df, dict_parameters)
            dict_parameters = self.add_to_dic("uniform_dis_mean_end", df, dict_parameters)
            dict_parameters = self.add_to_dic("uniform_dis_std_start", df, dict_parameters)
            dict_parameters = self.add_to_dic("uniform_dis_std_end", df, dict_parameters)
            
            self.distributions.append(dict_parameters)
            
       
    def __call__(self, image_info):
        
        # This is the main function. 
        self.augmentation_randomness_seed = 1024
        self.brain_image = image_info["brain_image"] 
        self.mask = image_info["mask"]
        self.gmm_adr = image_info["gmm_adr"] 
        self.scanner_index = image_info["scanner_index"]

        # Select distributions for scanners. 
        self.std_means_start = self.distributions[self.scanner_index]["uniform_dis_mean_start"]
        self.std_sigma_start = self.distributions[self.scanner_index]["uniform_dis_std_start"]
        self.std_means_end = self.distributions[self.scanner_index]["uniform_dis_mean_end"]
        self.std_sigma_end = self.distributions[self.scanner_index]["uniform_dis_std_end"]

        return self.generate_gmm_image()        
        
        
    def generate_gmm_image(self):
        """
        Augment the 3D image and get the slice. 
        """

        # 1. Mask and flatten brain
        masked_image = self.brain_image * self.mask
        data = masked_image[self.mask > 0]
        x = np.expand_dims(data, 1)
        
        # 2. Load the gmm model of the brain 
        gmm = pickle.load(open(self.gmm_adr, 'rb'))
        sort_indices = gmm.means_[:, 0].argsort(axis=0)

        probas_original = gmm.predict_proba(x)[:, sort_indices]

        # 3. Get the new intensity components
        params_dict = self.get_new_components(gmm, p_mu=None, q_sigma=None,
                                     std_means_start= self.std_means_start, std_means_end= self.std_means_end,
                                     std_sigma_start = self.std_means_start, std_sigma_end = self.std_means_end,
                                      range_seed = self.augmentation_randomness_seed)                             
        
        # 4. Get intensity of one voxels form the three tissue type distributions. 
        intensities_im = self.reconstruct_intensities(data, params_dict)
        
        # 5. Then we add the three predicted images by taking into consideration the probability that each pixel belongs to a
        # certain component of the gaussian mixture (probas_original)
        new_image_composed = self.get_new_image_composed(intensities_im, probas_original)
        
        
        # 7. Return the image in [0,1] (I changed it to a shift to have non-negative values).  
        new_image_composed = self.Shift_images(new_image_composed)
        
        # 6. Reconstruct the image to 3D. 
        new_image = np.zeros(self.brain_image.shape)
        new_image[np.where(self.mask > 0)] = new_image_composed


        # Save the augmented image
        return new_image
    

    def get_new_components(self, gmm, p_mu=None, q_sigma=None,
                       std_means_start=None, std_means_end=None, 
                       std_sigma_start=None, std_sigma_end=None,
                       range_seed=1024):

        sort_indices = gmm.means_[:, 0].argsort(axis=0)
        mu = np.array(gmm.means_[:, 0][sort_indices])
        std = np.array(np.sqrt(gmm.covariances_[:, 0])[sort_indices])
    
        n_components = mu.shape[0]
        # use pre-computed intervals to draw values for each component in the mixture
        if std_means_start is not None:
            # Set seeds for the random variables. 
            r1 = random.randint(20000, 30000)
            rng = np.random.default_rng(seed = r1)
            if p_mu is None:
                var_mean_diffs_start = np.array(std_means_start)
                var_mean_diffs_end = np.array(std_means_end)
                p_mu = rng.uniform(var_mean_diffs_start, var_mean_diffs_end)
        if std_sigma_start is not None:
            r1 = random.randint(1, 10000)
            rng = np.random.default_rng(seed = r1)
            if q_sigma is None:
                var_std_diffs_start = np.array(std_sigma_start)
                var_std_diffs_end = np.array(std_sigma_end)
                q_sigma = rng.uniform(var_std_diffs_start, var_std_diffs_end)
        else:
            # Draw random values for each component in the mixture
            # Multiply by random int for shifting left (-1), right (1) or not changing (0) the parameter.
            if p_mu is None:
                p_mu = 0.06 * np.random.random(n_components) * np.random.randint(-1, 2, n_components)
            if q_sigma is None:
                q_sigma = 0.005 * np.random.random(n_components) * np.random.randint(-1, 2, n_components)

        new_mu = mu + p_mu
        new_std = std + q_sigma
        return {'mu': mu, 'std': std, 'new_mu': new_mu, 'new_std': new_std}



    def reconstruct_intensities(self, data, dict_parameters):
        mu, std = dict_parameters['mu'], dict_parameters['std']
        new_mu, new_std = dict_parameters['new_mu'], dict_parameters['new_std']
        n_components = len(mu)
    
        # if we know the values of mean (mu) and standard deviation (sigma) we can find the new value of a voxel v
        # Fist we find the value of a factor w that informs about the percentile a given pixel belongs to: mu*v = d*sigma
        d_im = np.zeros(((n_components,) + data.shape))
        for k in range(n_components):
            d_im[k] = (data.ravel() - mu[k]) / (std[k] + 1e-7) # This is d_vk in the paper. 

        # we force the new pixel intensity to lie within the same percentile in the new distribution as in the original
        # distribution: px = mu + d*sigma
        intensities_im = np.zeros(((n_components,) + data.shape))
        for k in range(n_components):
            intensities_im[k] = new_mu[k] + d_im[k] * new_std[k] # This is Equation (3) in the paper. 

        return intensities_im
    
    
    def get_new_image_composed(self, intensities_im, probas_original):
        n_components = probas_original.shape[1]
        new_image_composed = np.zeros(intensities_im[0].shape)
        for k in range(n_components):
            new_image_composed = new_image_composed + probas_original[:, k] * intensities_im[k]

        return new_image_composed 
    
    
    
    def Shift_images(self, image):
    
        Min = image.min()
        if (Min < 0):
            image = image - Min
        return image
