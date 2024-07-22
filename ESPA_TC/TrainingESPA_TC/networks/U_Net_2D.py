import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.Linear_layer import Linear


class Down_module(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(Down_module, self).__init__()
        self.in_channels = in_channels 
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        
        self.Conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, padding= self.padding)
        self.batchN1 = nn.BatchNorm2d(self.out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.Conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=self.kernel_size, padding= self.padding)
        self.batchN2 = nn.BatchNorm2d(self.out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.Maxpool = nn.MaxPool2d(2)


    def forward(self, x):
        
        x = self.Conv1(x)
        x =  self.batchN1(x)
        x = self.relu1(x)
        x = self.Conv2(x)
        x = self.batchN2(x)
        x = self.relu2(x)
        return x
    
class Up_module(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(Up_module, self).__init__()
        self.in_channels = in_channels 
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        
        self.Upsample1 = nn.Upsample(scale_factor=2)
        self.Conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, padding= self.padding)
        
        self.Conv2 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, padding= self.padding)
        self.batchN2 = nn.BatchNorm2d(self.out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.Conv3 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=self.kernel_size, padding= self.padding)
        self.batchN3 = nn.BatchNorm2d(self.out_channels)
        self.relu3 = nn.ReLU(inplace=True)



    def forward(self, x, concatenated_layer):
        
        x = self.Upsample1(x)
        x = self.Conv1(x)
        x = torch.cat((concatenated_layer, x), dim = 1)

        x = self.Conv2(x)
        x = self.batchN2(x)
        x = self.relu2(x)
        
        x = self.Conv3(x)
        x = self.batchN3(x)
        x = self.relu3(x)
        return x
    
class generate_embedding(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(generate_embedding, self).__init__()
        self.in_channels = in_channels 
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        self.conv1 = nn.Conv2d(in_channels  = self.in_channels, out_channels = self.out_channels, kernel_size= self.kernel_size, padding = self.padding)
        self.batch1 = nn.BatchNorm2d(self.out_channels)
        self.Relu1 = nn.ReLU(inplace=True)    
        
    def forward(self, x):
        
        x =  self.conv1(x)
        x = self.batch1(x)
        x = self.Relu1(x)
        return x
    
       
class UNet_2D(nn.Module):
    
    def __init__(self, n_channels):
        
        super(UNet_2D, self).__init__()
        
        self.n_channels = n_channels # put other variables in here. 
        self.Down1 = Down_module(1, 32, 3, 1)
        self.Down2 = Down_module(32, 64, 3, 1)
        self.Down3 = Down_module(64, 128, 3, 1)
    
        self.Maxpool1 = nn.MaxPool2d(2)
        self.Maxpool2 = nn.MaxPool2d(2)
        
        self.Up1 = Up_module(128, 64, 3, 1)
        self.Up2 = Up_module(64, 32, 3, 1)
        
        self.generate_embedding1 = generate_embedding(32, self.n_channels, 3, 1)
        
     
    def forward(self, x):
        
        # Down1
        x = self.Down1(x)
        layer1_for_concatenation = x
        x = self.Maxpool1(x)
        
        # Down2
        x = self.Down2(x)
        layer2_for_concatenation = x
        x = self.Maxpool2(x)
        
        # Embedding
        x = self.Down3(x)
        
        # Up1
        x = self.Up1(x, layer2_for_concatenation)
        
        # Up2
        x = self.Up2(x, layer1_for_concatenation)
        
        # generate embedding
        x = self.generate_embedding1(x)

        return x

   

