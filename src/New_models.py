'''
Created on Oct 1, 2023

@author: MAE82
'''
import torch
import torch.nn as nn

'''
class Generator(nn.Module):
    def __init__(self, z_dim, filters, img_size_x, img_size_y):
        super(Generator, self).__init__()
        
        ######################### variables #########################
        self.z_dim = z_dim
        self.filters = filters
        self.img_size_x = img_size_x
        self.img_size_y = img_size_y 
        self.hid_dim = int(self.img_size_x * self.img_size_y * self.filters[4]/(32*32))
        self.latent_dim = 128
        self.z_hid_dim = int(self.img_size_x*self.img_size_y * self.filters[4]/(32*32))
        self.dim_x = int(self.img_size_x/32)
        self.scale_factor = 2
        
        ######################### layers #########################
        self.linear1 = nn.Linear(in_features = self.z_dim, out_features = self.z_hid_dim, bias=True)
        
        
    def forward(self, z, image):
        
        ########################################## part of network for z ############################################    
            # layer1: FCN, Relu, Reshaping
            x = nn.Linear(in_features = self.z_dim, out_features = self.z_hid_dim, bias=True)(z)
            x = nn.ReLU(inplace=True)(x)
            x = torch.reshape(x, (x.shape[0], self.filters[4], self.dim_x, self.dim_x))
            
            # layer2: upsampling, conv2D, Batch normalization, Relu
            x = nn.Upsample(scale_factor = self.scale_factor, mode='bilinear')(x)
            x = nn.Conv2d(in_channels = x.shape[1], out_channels = self.filters[4], kernel_size = (3, 3), stride = (1, 1), padding = 1)(x)
            x = nn.BatchNorm2d(x.shape[1])(x)
            x = nn.ReLU(True)(x)
            
            # layer3: upsampling, conv2D, Batch normalization, Relu
            x = nn.Upsample(scale_factor = self.scale_factor, mode='bilinear')(x)
            x = nn.Conv2d(in_channels = x.shape[1], out_channels = self.filters[3], kernel_size = (3, 3), stride = (1, 1), padding = 1)(x)
            x = nn.BatchNorm2d(x.shape[1])(x)
            x = nn.ReLU(True)(x)
            
            # layer4: upsampling, conv2D, Batch normalization, Relu
            x = nn.Upsample(scale_factor = self.scale_factor, mode='bilinear')(x)
            x = nn.Conv2d(in_channels = x.shape[1], out_channels = self.filters[2], kernel_size = (3, 3), stride = (1, 1), padding = 1)(x)
            x = nn.BatchNorm2d(x.shape[1])(x)
            x = nn.ReLU(True)(x)
            
            # layer3: upsampling, conv2D, Batch normalization, Relu
            x = nn.Upsample(scale_factor = self.scale_factor, mode='bilinear')(x)
            x = nn.Conv2d(in_channels = x.shape[1], out_channels = self.filters[1], kernel_size = (3, 3), stride = (1, 1), padding = 1)(x)
            x = nn.BatchNorm2d(x.shape[1])(x)
            x = nn.ReLU(True)(x)
            
            # layer4: upsampling, conv2D, Batch normalization, Relu
            x = nn.Upsample(scale_factor = self.scale_factor, mode='bilinear')(x)
            x = nn.Conv2d(in_channels = x.shape[1], out_channels = self.filters[1], kernel_size = (3, 3), stride = (1, 1), padding = 1)(x)
            x = nn.BatchNorm2d(x.shape[1])(x)
            x = nn.ReLU(True)(x)
            
        ########################################## part of network for image ############################################   
            image = nn.Conv2d(in_channels = image.shape[1], out_channels = self.filters[1], kernel_size = (3, 3), stride = (1, 1), padding = 1)(image)
            image = nn.BatchNorm2d(image.shape[1])(image)
            
            image = nn.Conv2d(in_channels = image.shape[1], out_channels = self.filters[1], kernel_size = (3, 3), stride = (1, 1), padding = 1)(image)
            image = nn.BatchNorm2d(image.shape[1])(image)
            
            cat = torch.cat((x, image), dim = 1)
            
        ####################################### Mutual network for z and image #########################################
            cat = nn.Conv2d(in_channels = cat.shape[1], out_channels = self.filters[1], kernel_size = (3, 3), stride = (1, 1), padding = 1)(cat)
            cat = nn.BatchNorm2d(cat.shape[1])(cat)
            cat = nn.ReLU(True)(cat)
            
            cat = nn.Conv2d(in_channels = cat.shape[1], out_channels = self.filters[1], kernel_size = (3, 3), stride = (1, 1), padding = 1)(cat)
            cat = nn.BatchNorm2d(cat.shape[1])(cat)
            cat = nn.ReLU(True)(cat)
            
            cat = nn.Conv2d(in_channels = cat.shape[1], out_channels = self.filters[0], kernel_size = (1, 1), stride = (1, 1), bias = False)(cat)
            cat = nn.Tanh()(cat)
            image = torch.add(cat, image)

            return image
'''


class Generator(nn.Module):
    def __init__(self, z_dim, filters, img_size_x, img_size_y):
        super(Generator, self).__init__()
        
        ######################### variables #########################
        self.z_dim = z_dim
        self.filters = filters
        self.img_size_x = img_size_x
        self.img_size_y = img_size_y 
        self.hid_dim = int(self.img_size_x * self.img_size_y * self.filters[4]/(32*32))
        self.latent_dim = 128
        self.z_hid_dim = int(self.img_size_x*self.img_size_y * self.filters[4]/(32*32))
        self.dim_x = int(self.img_size_x/32)
        self.scale_factor = 2
        
        ######################### layers: z part #########################
        self.linear_L1 = nn.Linear(in_features = self.z_dim, out_features = self.z_hid_dim, bias=True)
        self.conv_L2 = nn.Conv2d(in_channels = self.filters[4], out_channels = self.filters[4], kernel_size = (3, 3), stride = (1, 1), padding = 1)
        self.batch_L2 = nn.BatchNorm2d(self.filters[4])
        self.conv_L3 = nn.Conv2d(in_channels = self.filters[4], out_channels = self.filters[3], kernel_size = (3, 3), stride = (1, 1), padding = 1)
        self.batch_L3 = nn.BatchNorm2d(self.filters[3])
        self.conv_L4 = nn.Conv2d(in_channels = self.filters[3], out_channels = self.filters[2], kernel_size = (3, 3), stride = (1, 1), padding = 1)
        self.batch_L4 = nn.BatchNorm2d(self.filters[2])
        self.conv_L5 = nn.Conv2d(in_channels = self.filters[2], out_channels = self.filters[1], kernel_size = (3, 3), stride = (1, 1), padding = 1)
        self.batch_L5 = nn.BatchNorm2d(self.filters[1])
        self.conv_L6 = nn.Conv2d(in_channels = self.filters[1], out_channels = self.filters[1], kernel_size = (3, 3), stride = (1, 1), padding = 1)
        self.batch_L6 = nn.BatchNorm2d(self.filters[1])
        
        ######################### layers: image part #########################
        self.conv_image_L1  = nn.Conv2d(in_channels = self.filters[0], out_channels = self.filters[1], kernel_size = (3, 3), stride = (1, 1), padding = 1)
        self.batch_image_L1 = nn.BatchNorm2d(self.filters[1])
        self.conv_image_L2  = nn.Conv2d(in_channels = self.filters[1], out_channels = self.filters[1], kernel_size = (3, 3), stride = (1, 1), padding = 1)
        self.batch_image_L2 = nn.BatchNorm2d(self.filters[1])
        
        ######################### layers: Mutual part #########################
        self.conv_mutual_L1 = nn.Conv2d(in_channels = self.filters[1] * 2, out_channels = self.filters[1], kernel_size = (3, 3), stride = (1, 1), padding = 1)
        self.batch_mutual_L1 = nn.BatchNorm2d(self.filters[1])            
        self.conv_mutual_L2 = nn.Conv2d(in_channels = self.filters[1], out_channels = self.filters[1], kernel_size = (3, 3), stride = (1, 1), padding = 1)
        self.batch_mutual_L2 = nn.BatchNorm2d(self.filters[1])            
        self.conv_mutual_L3 = nn.Conv2d(in_channels = self.filters[1], out_channels = self.filters[0], kernel_size = (1, 1), stride = (1, 1), bias = False)
            
                
    def forward(self, z, original_image):
        
        ########################################## part of network for z ############################################    
            # layer1: FCN, Relu, Reshaping
            x = self.linear_L1(z)
            x = nn.ReLU(inplace=True)(x)
            x = torch.reshape(x, (x.shape[0], self.filters[4], self.dim_x, self.dim_x))
            
            # layer2: upsampling, conv2D, Batch normalization, Relu
            x = nn.Upsample(scale_factor = self.scale_factor, mode='bilinear')(x)
            x = self.conv_L2(x)
            x = self.batch_L2(x)
            x = nn.ReLU(inplace=True)(x)
            
            # layer3: upsampling, conv2D, Batch normalization, Relu
            x = nn.Upsample(scale_factor = self.scale_factor, mode='bilinear')(x)
            x = self.conv_L3(x)
            x = self.batch_L3(x)
            x = nn.ReLU(inplace=True)(x)
            
            # layer4: upsampling, conv2D, Batch normalization, Relu
            x = nn.Upsample(scale_factor = self.scale_factor, mode='bilinear')(x)
            x = self.conv_L4(x)
            x = self.batch_L4(x)
            x = nn.ReLU(inplace=True)(x)
            
            # layer3: upsampling, conv2D, Batch normalization, Relu
            x = nn.Upsample(scale_factor = self.scale_factor, mode='bilinear')(x)
            x = self.conv_L5(x)
            x = self.batch_L5(x)
            x = nn.ReLU(inplace=True)(x)
            
            # layer4: upsampling, conv2D, Batch normalization, Relu
            x = nn.Upsample(scale_factor = self.scale_factor, mode='bilinear')(x)
            x = self.conv_L6(x)
            x = self.batch_L6(x)
            x = nn.ReLU(inplace=True)(x)
            
        ########################################## part of network for image ############################################   
            image = self.conv_image_L1(original_image)
            image = self.batch_image_L1(image)
            image = self.conv_image_L2(image)
            image = self.batch_image_L2(image)
            cat = torch.cat((x, image), dim = 1)
            
            
        ####################################### Mutual network for z and image #########################################
            cat = self.conv_mutual_L1(cat)
            cat = self.batch_mutual_L1(cat)
            cat = nn.ReLU(inplace=True)(cat)
            
            cat = self.conv_mutual_L2(cat)
            cat = self.batch_mutual_L2(cat)
            cat = nn.ReLU(inplace=True)(cat)
            
            cat = self.conv_mutual_L3(cat)
            cat = nn.Tanh()(cat)
            Final_image = torch.add(cat, original_image)

            return Final_image, cat

        
class Discriminator(nn.Module):
    def __init__(self, filters, image_x_dim, image_y_dim):
        super(Discriminator, self).__init__()
        self.filters = filters
        self.LR_negative_slope = 0.2
        self.hid_dim = int(image_x_dim * image_y_dim * self.filters[4]/(32*32))
        
        ########################################## layers for network ############################################ 
        self.conv_L1 = nn.Conv2d(in_channels = self.filters[0], out_channels = self.filters[1], kernel_size = (5, 5), stride = (2, 2), padding = [2, 2])
        self.batch_L1 = nn.BatchNorm2d(self.filters[1])
        
        self.conv_L2 = nn.Conv2d(in_channels = self.filters[1], out_channels = self.filters[2], kernel_size = (5, 5), stride = (2, 2), padding = [2, 2])
        self.batch_L2 = nn.BatchNorm2d(self.filters[2])
        
        self.conv_L3 = nn.Conv2d(in_channels = self.filters[2], out_channels = self.filters[3], kernel_size = (5, 5), stride = (2, 2), padding = [2, 2])
        self.batch_L3 = nn.BatchNorm2d(self.filters[3])
        
        self.conv_L4 = nn.Conv2d(in_channels = self.filters[3], out_channels = self.filters[4], kernel_size = (5, 5), stride = (2, 2), padding = [2, 2])
        self.batch_L4 = nn.BatchNorm2d(self.filters[4])
        
        self.conv_L5 = nn.Conv2d(in_channels = self.filters[4], out_channels = self.filters[4], kernel_size = (5, 5), stride = (2, 2), padding = [2, 2])
        self.batch_L5 = nn.BatchNorm2d(self.filters[4])
        
        self.linear_L6 = nn.Linear(in_features = self.hid_dim , out_features = self.filters[4], bias=True)
        self.linear_L7 = nn.Linear(in_features = self.filters[4], out_features = self.filters[0], bias=True)
        
    def forward(self, orig_image):
        
        # layer1: conv2D, Batch normalization, Leaky Relu
        image = self.conv_L1(orig_image)
        image = self.batch_L1(image)
        image = nn.LeakyReLU(self.LR_negative_slope)(image)
        
        # layer2: conv2D, Batch normalization, Leaky Relu
        image = self.conv_L2(image)
        image = self.batch_L2(image)
        image = nn.LeakyReLU(self.LR_negative_slope)(image)
        
        # layer3: conv2D, Batch normalization, Leaky Relu
        image = self.conv_L3(image)
        image = self.batch_L3(image)
        image = nn.LeakyReLU(self.LR_negative_slope)(image)
        
        # layer4: conv2D, Batch normalization, Leaky Relu
        image = self.conv_L4(image)
        image = self.batch_L4(image)
        image = nn.LeakyReLU(self.LR_negative_slope)(image)
        
        # layer5: conv2D, Batch normalization, Leaky Relu
        image = self.conv_L5(image)
        image = self.batch_L5(image)
        image = nn.LeakyReLU(self.LR_negative_slope)(image)
        
        # layer6: Flattening layer
        image = nn.Flatten()(image)
        
        # layer7: FCN, Leaky Relu
        image = self.linear_L6(image)
        image = nn.LeakyReLU(self.LR_negative_slope)(image)
        
        # layer8: FCN, Leaky Relu
        prediction = self.linear_L7(image)
        prediction = nn.Sigmoid()(prediction)
        
        return prediction
         
