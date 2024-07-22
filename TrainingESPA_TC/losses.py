"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function
import os 
import torch
import torch.nn as nn
import random
import numpy as np

seed = 1024


def set_seeds():
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seeds()



class MISPEL_losses(nn.Module):
    
    def __init__(self, step_number, lambda1, lambda2, lambda3, lambda4, batch_size, image_number): #*** Mah
        
        super(MISPEL_losses, self).__init__()
        self.step_number = step_number
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        self.batch_size = batch_size
        self.sum = 0
        self.MAE = nn.L1Loss()
        self.image_number = image_number #*** Mah

    
    def l_coulpling(self, embedding_sets):
        ''' Extracting loss for all images of batch '''
        
        # 
        for ind in range(0, len(embedding_sets)):
            embedding_sets[ind] = embedding_sets[ind].flatten().unsqueeze(dim = 1)

        concatenated_embeddings = torch.cat(embedding_sets, dim = 1)
        variances = concatenated_embeddings.var(dim = 1)
        l_coulpling = variances.mean()
        return l_coulpling
    
    
    def l_reconstruction(self, original_images, reconstructed_images):
        
        # Sigma_i=1:M(MAE(x_i^j, xBar_i^j)) 
        self.sum = 0.0
        for i in range(0, original_images.shape[0]):
            self.sum += self.MAE(original_images[i, ...], reconstructed_images[i, ...])
            
        # This is average loss for each batch. 
        return (self.sum)/(self.batch_size)

    
    def l_harmonization(self, reconstructed_image_set):
        
        self.sum = 0.0
        
        for i in range(0, len(reconstructed_image_set[0])):
            for j in range(0, self.image_number):
                for k in range(j + 1, self.image_number):
                    self.sum += self.MAE(reconstructed_image_set[j][i], reconstructed_image_set[k][i])
        permutation_no = 2/((self.image_number) * (self.image_number - 1))

        return self.sum/((permutation_no) * (self.batch_size))
    
    
    def forward(self, original_images, embeddings, reconstructed_images):
        
        original_images = torch.cat(original_images, dim=0) 
        reconstructed_images_concatenated = torch.cat(reconstructed_images, dim=0) 
            
        # compute l_reconstruction0
        l_reocn = self.l_reconstruction(original_images, reconstructed_images_concatenated)
            
        if self.step_number == 1:
            # compute l_coulpling
            l_coup = self.l_coulpling(embeddings) 
            loss_step1 = self.lambda1 * l_reocn + self.lambda2 * l_coup  
            return loss_step1, l_reocn, l_coup     

        if self.step_number == 2:
            # compute l_harmonization
            l_harm = self.l_harmonization(reconstructed_images)
            loss_step2 = self.lambda3 * l_reocn + self.lambda4 * l_harm
            return loss_step2, l_reocn, l_harm

        else: 
            print("Invalid number for taring steps!")
            return 0.0

