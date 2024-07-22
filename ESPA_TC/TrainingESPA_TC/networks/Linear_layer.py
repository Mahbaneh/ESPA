'''
Created on May 18, 2022

@author: MAE82
'''
import torch
import torch.nn as nn
import math


class Linear(nn.Module):
    
    def __init__(self, embedding_length):
        
        super().__init__()
        self.embedding_length = embedding_length
        self.weights = nn.Parameter(torch.Tensor(1, self.embedding_length))
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))

    
    def forward(self, x):

        x = x *  self.weights[..., None, None]
        x = x.sum(dim = 1)
        return x

