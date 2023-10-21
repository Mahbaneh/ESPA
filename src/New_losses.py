'''
Created on Oct 6, 2023

@author: MAE82
'''

import torch.nn as nn
from torch import linalg as LA
import torch

class losses(nn.Module):
    
    def __init__(self): 
        super(losses, self).__init__()
        self.criterion = nn.BCELoss()
        
    
    def l_disc(self, images, label, loss_lambda):
        return loss_lambda * self.criterion(images, label)
    
    
    def l_gen(self, images, label, loss_lambda):
        return loss_lambda * self.criterion(images, label)

    
    def l_ID(self, residuals, loss_lambda):
        '''
        sum = 0.0
        for indi in range()
        torch.norm(residuals, dim = ).view(-1)
        '''
        residuals = torch.squeeze(residuals)
        return torch.mul(torch.mean(LA.matrix_norm(residuals)), -1.0 * loss_lambda) 

    
    
    def forward(self, network_type, loss_lambda, images, label):
        
        if network_type == "gen":
            return self.l_gen(images, label, loss_lambda)

        elif network_type == "dsc":
            return self.l_disc(images, label, loss_lambda)
            
        else:
            # network_type == "ID"
            return self.l_ID(images, loss_lambda)

        
        
        