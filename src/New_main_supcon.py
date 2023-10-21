import os
import argparse
import torch
import random
import numpy as np
from os.path import exists
from DataSets import ScannerDataset, ScannerDataset_withsliceIndex
'''
from New_utils import read_datasets, weights_init, save_model, plot_losses_prob,\
 plot_generated_images, animating_images, plot_external_residual_generated_image,\
plot_external_residual_generated_image2, plot_all_images
'''
from New_utils import read_datasets, weights_init, save_model, plot_losses_prob,\
plot_external_residual_generated_image2, plot_all_images, plot_all_images_nifti,\
plot_external_residual_generated_image_nifti, read_datasets_withsliceIndex,\
 generate_masked_images, reshape_images
import matplotlib.pyplot as plt
from New_models import Generator, Discriminator
import torch.nn as nn
import torch.optim as optim
from New_losses import losses
import torchvision.utils as vutils
import matplotlib.animation as animation
import nibabel as nib


def set_seeds(seed1):
    random.seed(seed1)
    np.random.seed(seed1)
    torch.manual_seed(seed1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def set_seeds_torch(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seeds(1024)


def parse_option(gpu_number):
    
    ####################################### I need ###########################################
    parser = argparse.ArgumentParser('argument for training')
    
    parser.add_argument('--checkpoint_plot_frequency', type=int, default= 1,
                        help='checkpoint to save models and outputs')
    
    parser.add_argument('--checkpoint_model_frequency', type=int, default= 5,
                        help='checkpoint to save models and outputs')
    
    parser.add_argument('--num_workers', type=int, default= 0,
                        help='No of workers to use')
    
    parser.add_argument('--batch_size', type=int, default= 20,
                        help='Batch_size')
    
    parser.add_argument('--target_scanner_name', type=str, default= "ge",
                        help='Name of the target scanner.') 
    
    parser.add_argument('--mask_adr', type=str, default= \
                         "/Users/MAE82/eclipse-workspace/00_Aug_Residual_SimpleGAN_cleaned/src/Dataset/cropped_JHU_MNI_SS_T1_Brain_Mask.nii.gz", \
                        help='Brain mask address.') 
    
    parser.add_argument('--external_scanner_name', type=str, default= "scanner1",
                        help='Name of the target scanner.') 
    
    parser.add_argument('--epochs', type=int, default=1000,
                        help='No of the training epochs')
    
    parser.add_argument('--lambda_ID', type=float, default = 0.001,
                        help='Weight of identity_loss')
    
    parser.add_argument('--lambda_gen', type=float, default = 1.0,
                        help='Weight of generator_loss')
    
    parser.add_argument('--lambda_dsc', type=float, default = 1.0,
                        help='Weight of discriminator_loss')
    
    parser.add_argument('--external_scanner_image_adrs', type=str, default="./Dataset/ExternalScanner",
                        help='')
    
    parser.add_argument('--target_scanner_image_adrs', type=str, default="./Dataset/TargetScanners",
                        help='Adr to the target scanners directory')
    
    parser.add_argument('--CVfolds_adrs', type=str, default="./Dataset/CV_Folds",
                        help='Adr to the excel file of folds')
    
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Adam Learning rate')
    
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Adam beta1')
    
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Adam beta2')
    
    parser.add_argument('--CV_no', type=int, default= None,
                        help='Index of the fold for cross validation')
    
    parser.add_argument('--z_dim', type=int, default= 100,
                        help='Size of z latent vector')
    
    parser.add_argument('--image_x_dim', type=int, default= 224,
                        help='first dimension of the image')
    
    parser.add_argument('--image_y_dim', type=int, default= 224,
                        help='first dimension of the image')
    
    ########################################## I may need ########################################
    parser.add_argument('--no_axial_slices', type=int, default=156,
                        help='No of axial slices in each scan')

    opt = parser.parse_args()
    
    ################################### making directories ###################################
    opt.saving_dir = os.path.join("./", "save" + str(gpu_number))
    if (exists(opt.saving_dir) == False):
        os.mkdir(opt.saving_dir)
        
    opt.Results = os.path.join(opt.saving_dir, "Results")
    if (exists(opt.Results) == False):
        os.mkdir(opt.Results)
        
    opt.Networks = os.path.join(opt.saving_dir, "Networks")
    if (exists(opt.Networks) == False):
        os.mkdir(opt.Networks)
        
    opt.models = os.path.join(opt.Networks, "models")
    if (exists(opt.models) == False):
        os.mkdir(opt.models)
        
    opt.tensorboard = os.path.join(opt.Networks, "tensorboard")
    if (exists(opt.tensorboard) == False):
        os.mkdir(opt.tensorboard)
        
    opt.model_name = 'model_{}_bsz_{}_epo_{}_lr_{}_ID_{}_gen_{}_dsc_{}_CV_no_{}'\
    .format(opt.target_scanner_name, opt.batch_size, opt.epochs, opt.learning_rate,\
            opt.lambda_ID, opt.lambda_gen, opt.lambda_dsc, opt.CV_no)
    opt.model_adr = os.path.join(opt.models, opt.model_name)
    if (exists(opt.model_adr) == False):
        os.mkdir(opt.model_adr)

    opt.tb_adr = os.path.join(opt.tensorboard, opt.model_name)
    if (exists(opt.tb_adr) == False):
        os.mkdir(opt.tb_adr)
        
    return opt

    
def set_loaders(opt, Target_scanner_images, External_scanner_images,\
                 Target_scanner_indices, External_scanner_indices):
    
    # Loading the dataset. 
    Target_scanner_dataset = ScannerDataset(Target_scanner_images, Target_scanner_indices)
    External_scanner_dataset = ScannerDataset(External_scanner_images, External_scanner_indices)
    
    Target_scanner_loader = torch.utils.data.DataLoader(
        Target_scanner_dataset, batch_size = opt.batch_size, shuffle = True,
        num_workers = opt.num_workers, pin_memory = True, sampler = None)
    
    External_scanner_loader = torch.utils.data.DataLoader(
        External_scanner_dataset , batch_size = opt.batch_size, shuffle = True,
        num_workers = opt.num_workers, pin_memory = True, sampler = None)
    
    return  Target_scanner_loader, External_scanner_loader



def set_loaders_withsliceIndex(opt, Target_scanner_images, External_scanner_images,\
                 Target_scanner_indices, External_scanner_indices):
    
    # Loading the dataset. 
    Target_scanner_dataset = ScannerDataset_withsliceIndex(Target_scanner_images, Target_scanner_indices)
    External_scanner_dataset = ScannerDataset_withsliceIndex(External_scanner_images, External_scanner_indices)
    
    Target_scanner_loader = torch.utils.data.DataLoader(
        Target_scanner_dataset, batch_size = opt.batch_size, shuffle = True,
        num_workers = opt.num_workers, pin_memory = True, sampler = None)
    
    External_scanner_loader = torch.utils.data.DataLoader(
        External_scanner_dataset , batch_size = opt.batch_size, shuffle = True,
        num_workers = opt.num_workers, pin_memory = True, sampler = None)
    
    return  Target_scanner_loader, External_scanner_loader  


def set_model(opt, device, filters):
    
    # Build models
    netG = Generator(opt.z_dim, filters, opt.image_x_dim, opt.image_y_dim).to(device)
    netG.apply(weights_init)
    
    netD = Discriminator(filters, opt.image_x_dim, opt.image_y_dim).to(device)
    netG.apply(weights_init)
    
    # Build criterion
    criterion = losses()
    criterion = criterion.to(device)
    
    # Build optimizers 
    optimizerD = optim.Adam(netD.parameters(), lr = opt.learning_rate, betas=(opt.beta1, opt.beta2))
    optimizerG = optim.Adam(netG.parameters(), lr = opt.learning_rate, betas=(opt.beta1, opt.beta2))
    
    return netG, netD, criterion, optimizerD, optimizerG 



def training(opt, device, Target_scanner_loader, External_scanner_loader, netG, netD, real_label,\
              fake_label, criterion, optimizerD, optimizerG, fixed_noise):
    
    ################## variables to save statistics ##################
    img_list = []
    G_losses = []
    D_losses = []
    D_losses_fake = []
    D_losses_real = []
    D_x_prob = []
    D_G_z1_prob = []
    D_G_z2_prob = []
    iters = 0
    
    ########################### training ############################
    for epoch in range(opt.epochs):
        ############### read all batched for external scanner ##########
        External_scanner_batches = []
        External_scanner_images = []
        
        for batch, images in enumerate(External_scanner_loader):
            External_scanner_batches.append(batch)
            External_scanner_images.append(images)
           
        if(epoch == 0):
            selected_images = External_scanner_images[0].clone()

        for batch, images in enumerate(Target_scanner_loader):
            #################### training discriminator ################
            # Discriminating the real data: images of the target scanner
            netD.zero_grad() # Erase gradient of parameters
            images = images.to(device)
            b_size = images.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = netD(images).view(-1)
            errD_real = criterion("dsc", opt.lambda_dsc, output, label) 
            errD_real.backward()
            D_x = output.mean().item()
            
            # We save the gradient by not calling netD.zero_grad(). 
            noise = torch.randn(b_size, opt.z_dim, device=device) #*** This is the noise. 
            fake, residual = netG(noise, External_scanner_images[batch].to(device))
            detached_residual = residual.clone().detach()
            label.fill_(fake_label) 
            output = netD(fake.detach()).view(-1) # Detach fake to not use netG for optimization.
            errD_fake = criterion("dsc", opt.lambda_dsc, output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step() # Updating weights based on the gradient
            
            ################# training generator #################
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake).view(-1)
            errG = criterion("gen", opt.lambda_gen, output, label)
            errID = criterion("ID", opt.lambda_ID, detached_residual, None)
            GtotalErr = errG + errID
            GtotalErr.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
            
            ######################## saving results ##########################
            if batch % 5 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, opt.epochs, batch, len(Target_scanner_loader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                
            
            G_losses.append(errG.item() / opt.lambda_gen)
            D_losses.append(errD.item() / opt.lambda_dsc)
            D_losses_fake.append(errD_fake.item() /opt.lambda_dsc)
            D_losses_real.append(errD_real.item()/ opt.lambda_dsc)
            
            D_x_prob.append(D_x)
            D_G_z1_prob.append(D_G_z1) 
            D_G_z2_prob.append(D_G_z2)
            
            evaluation(batch, Target_scanner_loader, epoch, opt, G_losses, D_losses, D_losses_fake, D_losses_real, \
                        D_x_prob, D_G_z1_prob, D_G_z2_prob, netG, optimizerG, netD, optimizerD, fixed_noise,\
                 selected_images, device)
       
            iters +=1
            
def training_withsliceIndex(opt, device, Target_scanner_loader, External_scanner_loader, netG, netD, real_label,\
              fake_label, criterion, optimizerD, optimizerG, fixed_noise, mask):
    
    ################## variables to save statistics ##################
    img_list = []
    G_losses = []
    D_losses = []
    D_losses_fake = []
    D_losses_real = []
    D_x_prob = []
    D_G_z1_prob = []
    D_G_z2_prob = []
    iters = 0
    
    ########################### training ############################
    for epoch in range(opt.epochs):
        ############### read all batched for external scanner ##########
        External_scanner_batches = []
        External_scanner_images = []
        External_scanner_indices = []
        
        for batch, [images, indices] in enumerate(External_scanner_loader):
            External_scanner_batches.append(batch)
            External_scanner_images.append(images)
            External_scanner_indices.append(indices)
           
        if(epoch == 0):
            selected_images = External_scanner_images[0].clone()

        for batch, [images, indices] in enumerate(Target_scanner_loader):
            #################### training discriminator ################
            # Discriminating the real data: images of the target scanner
            netD.zero_grad() # Erase gradient of parameters
            images = images.to(device)
            b_size = images.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = netD(images).view(-1)
            errD_real = criterion("dsc", opt.lambda_dsc, output, label) 
            errD_real.backward()
            D_x = output.mean().item()
            
            # We save the gradient by not calling netD.zero_grad(). 
            noise = torch.randn(b_size, opt.z_dim, device=device) #*** This is the noise. 
            fake, residual = netG(noise, External_scanner_images[batch].to(device))
            detached_residual = residual.clone().detach()
            detached_fake = fake.clone().detach()
            detached_fake = generate_masked_images(detached_fake, External_scanner_indices[batch], mask.detach())
            label.fill_(fake_label) 
            output = netD(detached_fake).view(-1) # Detach fake to not use netG for optimization.
            errD_fake = criterion("dsc", opt.lambda_dsc, output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step() # Updating weights based on the gradient
            
            ################# training generator #################
            netG.zero_grad()
            label.fill_(real_label)
            fake = generate_masked_images(fake, External_scanner_indices[batch], mask.detach())
            output = netD(fake).view(-1)
            errG = criterion("gen", opt.lambda_gen, output, label)
            errID = criterion("ID", opt.lambda_ID, detached_residual, None)
            GtotalErr = errG + errID
            GtotalErr.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
            
            ######################## saving results ##########################
            if batch % 5 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, opt.epochs, batch, len(Target_scanner_loader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                
            
            G_losses.append(errG.item() / opt.lambda_gen)
            D_losses.append(errD.item() / opt.lambda_dsc)
            D_losses_fake.append(errD_fake.item() /opt.lambda_dsc)
            D_losses_real.append(errD_real.item()/ opt.lambda_dsc)
            
            D_x_prob.append(D_x)
            D_G_z1_prob.append(D_G_z1) 
            D_G_z2_prob.append(D_G_z2)
            
            evaluation(batch, Target_scanner_loader, epoch, opt, G_losses, D_losses, D_losses_fake, D_losses_real, \
                        D_x_prob, D_G_z1_prob, D_G_z2_prob, netG, optimizerG, netD, optimizerD, fixed_noise,\
                 selected_images, device)
       
            iters +=1




def evaluation(batch, Target_scanner_loader, epoch, opt, G_losses, D_losses, D_losses_fake, D_losses_real, D_x_prob, \
                D_G_z1_prob, D_G_z2_prob, netG, optimizerG, netD, optimizerD, fixed_noise,\
                 selected_images, device):
    
    ###################################### checkpoint ############################################
    # Check how the generator is doing by saving G's output on fixed_noise
    if (batch == len(Target_scanner_loader) - 1):
        if ((epoch == opt.epochs - 1) or ((epoch + 1) % (opt.checkpoint_model_frequency) == 0)):
                    
            model_dir = os.path.join(opt.model_adr, "chkPoint_" + str(epoch + 1))
            if (exists(model_dir) == False):
                os.mkdir(model_dir)
            # Saving models
            save_model(netG, optimizerG, opt, epoch, os.path.join(model_dir, "Gen"))
            save_model(netD, optimizerD, opt, epoch, os.path.join(model_dir, "Dsc"))
            
                    
        if ((epoch == opt.epochs - 1) or ((epoch + 1) % (opt.checkpoint_plot_frequency) == 0)):
            model_dir = os.path.join(opt.model_adr, "chkPoint_" + str(epoch + 1))
            if (exists(model_dir) == False):
                os.mkdir(model_dir)
                        
            # Plotting losses and predictions
            plot_losses_prob(G_losses, D_losses, D_losses_fake, D_losses_real, D_x_prob, D_G_z1_prob, D_G_z2_prob, model_dir)
                    
            # Generated images
            with torch.no_grad():
                fake, res_images = netG(fixed_noise, selected_images.to(device)) 
                generated_fake = fake.clone().detach().cpu()
                generated_res_images = res_images.clone().detach().cpu()
                      
            #animating_images(generated_fake, model_dir)
            #plot_generated_images(generated_fake, generated_res_images, model_dir)
            
            plot_external_residual_generated_image2(selected_images, generated_fake, generated_res_images, model_dir)
            plot_all_images(selected_images, generated_fake, generated_res_images, model_dir)
            
            # nifti plots
            plot_external_residual_generated_image_nifti(selected_images, generated_fake, generated_res_images, model_dir)
            plot_all_images_nifti(selected_images, generated_fake, generated_res_images, model_dir)
            
               

def main(gpu_number):
       
    opt = parse_option(gpu_number)
    filters = [1, 16, 32, 64, 128, 256]
    real_label = 1.
    fake_label = 0.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fixed_noise = torch.randn(opt.batch_size, opt.z_dim, device=device)

    # Read data
    Target_scanner_images, External_scanner_images = read_datasets(opt)

    # Build data loader
    Target_scanner_loader, External_scanner_loader = set_loaders(opt, Target_scanner_images, \
                                                                   External_scanner_images)

    netG, netD, criterion, optimizerD, optimizerG = set_model(opt, device, filters)
    
    training(opt, device, Target_scanner_loader, External_scanner_loader, netG, netD, real_label,\
              fake_label, criterion, optimizerD, optimizerG, fixed_noise)
    
    print("Finished training!")
    


def main_with_sliceIndex(gpu_number):
       
    opt = parse_option(gpu_number)
    filters = [1, 16, 32, 64, 128, 256]
    real_label = 1.
    fake_label = 0.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fixed_noise = torch.randn(opt.batch_size, opt.z_dim, device=device)
    
    # Read mask
    mask = nib.load(opt.mask_adr).get_fdata()
    mask = reshape_images(mask, opt.image_x_dim,\
                                            opt.image_y_dim)
    mask = torch.from_numpy(mask)

    # Read data
    Target_scanner_images, External_scanner_images, Target_scanner_indices,\
     External_scanner_indices = read_datasets_withsliceIndex(opt)

    # Build data loader
    Target_scanner_loader, External_scanner_loader = set_loaders_withsliceIndex(opt, Target_scanner_images, \
                                                                   External_scanner_images, \
                                                                   Target_scanner_indices, \
                                                                   External_scanner_indices)

    netG, netD, criterion, optimizerD, optimizerG = set_model(opt, device, filters)
    
    training_withsliceIndex(opt, device, Target_scanner_loader, External_scanner_loader, netG, netD, real_label,\
              fake_label, criterion, optimizerD, optimizerG, fixed_noise, mask)
    
    print("Finished training!")

