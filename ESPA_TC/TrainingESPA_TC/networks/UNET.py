"""ResNet in PyTorch.
ImageNet-Style ResNet
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Adapted from: https://github.com/bearpaw/pytorch-classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.Linear_layer import Linear
from networks.U_Net_2D import UNet_2D


def UNET(embedding_no, **kwargs):
    return UNet_2D(embedding_no, **kwargs)

model_dict = {

    'UNET': UNET,
}

        
class UnSupConMISPEL(nn.Module):
    
    """U-Net + liners decoder"""
    
    def __init__(self, name = 'UNET', head = 'linear_decoder', latent_embedding_no = 6, scanner_no = 4): 
        super(UnSupConMISPEL, self).__init__()

        model_fun = model_dict[name]
        self.scanner_no = scanner_no
        self.encoders = nn.ModuleList([model_fun(latent_embedding_no) for i in range(scanner_no)])
        self.linears_decoders = nn.ModuleList([Linear(latent_embedding_no) for i in range(scanner_no)])
        
            
    def forward(self, x, tarining_step_number):
        
        '''
        I want the whole unit in MISPEL: (1) complete Unet, and (2) linear functions. 
        '''
        
        embeddings = []
        recontructed_images = []
        
        for i in range(self.scanner_no):
            embeddings.append(self.encoders[i](x[i]))
            recontructed_images.append(self.linears_decoders[i](embeddings[i]))

        return embeddings, recontructed_images   

