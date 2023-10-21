'''
Created on Oct 1, 2023

@author: MAE82
'''
import torch
import os  
import nibabel as nib
import pandas as pd
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable


def read_data_target_scanner(directory, scanner_name, CV_directory, CV_no):
    
    # Read excel file that contains the name of images
    if CV_no == None: 
        excel_file_adr = os.path.join(CV_directory, "Names_train.xlsx")
    else:
        excel_file_adr = os.path.join(CV_directory, "Names_train_fold" + str(CV_no) + ".xlsx")
    image_names = pd.read_excel(excel_file_adr)["Name_of_image"].values.tolist()
    subject_names = [image_name.replace("_UCDavis.nii.gz", "") for image_name in image_names]

    # Read images of target scanner 
    # Read the first image
    image_adr = os.path.join(directory, scanner_name, subject_names[0], image_names[0])
    stacked_slices = nib.load(image_adr).get_fdata()
       
    for indi in range(1, len(image_names)):
        image_adr = os.path.join(directory, scanner_name, subject_names[indi], image_names[indi])
        stacked_slices = np.concatenate([stacked_slices, nib.load(image_adr).get_fdata()],\
                                        axis = 2)
    return stacked_slices


def read_data_external_scanner(directory, scanner_name):
    
    # Read excel file that contains the name of images
    excel_file_adr = os.path.join(directory, scanner_name, "External_scanner_image_info.xlsx")
    image_names = pd.read_excel(excel_file_adr)["Name_of_image"].values.tolist()
    subject_names = pd.read_excel(excel_file_adr)["Subject_name"].values.tolist()
    
    # Read images of target scanner 
    # Read the first image
    image_adr = os.path.join(directory, scanner_name, subject_names[0], image_names[0])
    stacked_slices = nib.load(image_adr).get_fdata()
       
    for indi in range(1, len(image_names)):
        image_adr = os.path.join(directory, scanner_name, subject_names[indi], image_names[indi])
        stacked_slices = np.concatenate([stacked_slices, nib.load(image_adr).get_fdata()],\
                                        axis = 2)
    return stacked_slices


def read_data_target_scanner_withsliceIndex(directory, scanner_name, CV_directory, CV_no):
    
    # Read excel file that contains the name of images
    if CV_no == None: 
        excel_file_adr = os.path.join(CV_directory, "Names_train.xlsx")
    else:
        excel_file_adr = os.path.join(CV_directory, "Names_train_fold" + str(CV_no) + ".xlsx")
    image_names = pd.read_excel(excel_file_adr)["Name_of_image"].values.tolist()
    subject_names = [image_name.replace("_UCDavis.nii.gz", "") for image_name in image_names]

    # Read images of target scanner 
    # Read the first image
    image_adr = os.path.join(directory, scanner_name, subject_names[0], image_names[0])
    stacked_slices = nib.load(image_adr).get_fdata()
      
    indices = [ind for ind in range(0, stacked_slices.shape[2])] 
    for indi in range(1, len(image_names)):
        image_adr = os.path.join(directory, scanner_name, subject_names[indi], image_names[indi])
        data = nib.load(image_adr).get_fdata()
        stacked_slices = np.concatenate([stacked_slices, data],\
                                        axis = 2)
        indices = indices + [ind for ind in range(0, data.shape[2])]
    return stacked_slices, indices


def read_data_external_scanner_withsliceIndex(directory, scanner_name):
    
    # Read excel file that contains the name of images
    excel_file_adr = os.path.join(directory, scanner_name, "External_scanner_image_info.xlsx")
    image_names = pd.read_excel(excel_file_adr)["Name_of_image"].values.tolist()
    subject_names = pd.read_excel(excel_file_adr)["Subject_name"].values.tolist()
    
    # Read images of target scanner 
    # Read the first image
    image_adr = os.path.join(directory, scanner_name, subject_names[0], image_names[0])
    stacked_slices = nib.load(image_adr).get_fdata()
    
    indices = [ind for ind in range(0, stacked_slices.shape[2])]    
    for indi in range(1, len(image_names)):
        image_adr = os.path.join(directory, scanner_name, subject_names[indi], image_names[indi])
        data = nib.load(image_adr).get_fdata()
        stacked_slices = np.concatenate([stacked_slices, data],\
                                        axis = 2)
        indices = indices + [ind for ind in range(0, data.shape[2])]
    return stacked_slices, indices


def reshape_images2(images, x_add_dim, y_add_dim):
    
    # original dim of image is (152, 188)
    # I want to increase the size to (192, 192)
    # The (x_add_dim, y_add_dim) should be (20, 2)
    
    new_image = np.zeros((x_add_dim, y_add_dim, images.shape[2]))
    new_image[20:172, 2:190, :] = images
    return new_image


def reshape_images(images, x_add_dim, y_add_dim):
    
    # original dim of image is (152, 188)
    # I want to increase the size to (192, 192)
    # The (x_add_dim, y_add_dim) should be (20, 2)
    
    new_image = np.zeros((x_add_dim, y_add_dim, images.shape[2]))
    new_image[36:188, 18:206, :] = images
    return new_image



def read_datasets(opt):
    
    Target_scanner_images = read_data_target_scanner(opt.target_scanner_image_adrs,\
                              opt.target_scanner_name, opt.CVfolds_adrs, opt.CV_no)
    Target_scanner_images = reshape_images(Target_scanner_images, opt.image_x_dim,\
                                            opt.image_y_dim)
    
    External_scanner_images = read_data_external_scanner(opt.external_scanner_image_adrs, \
                                                         opt.external_scanner_name)
    External_scanner_images = reshape_images(External_scanner_images, opt.image_x_dim,\
                                              opt.image_y_dim)
    
    return Target_scanner_images, External_scanner_images



def read_datasets_withsliceIndex(opt):
    
    Target_scanner_images, Target_scanner_indices = read_data_target_scanner_withsliceIndex(opt.target_scanner_image_adrs,\
                              opt.target_scanner_name, opt.CVfolds_adrs, opt.CV_no)
    Target_scanner_images = reshape_images(Target_scanner_images, opt.image_x_dim,\
                                            opt.image_y_dim)
    
    External_scanner_images, External_scanner_images_indices = read_data_external_scanner_withsliceIndex(opt.external_scanner_image_adrs, \
                                                         opt.external_scanner_name)
    External_scanner_images = reshape_images(External_scanner_images, opt.image_x_dim,\
                                              opt.image_y_dim)
    
    return Target_scanner_images, External_scanner_images,\
         Target_scanner_indices, External_scanner_images_indices 



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias != None:
            nn.init.constant_(m.bias.data, 0)
            
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        

def save_model(model, optimizer, opt, epoch, save_file):
    #print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state
    
    
def plot_losses_prob(G_losses, D_losses, D_losses_fake, D_losses_real, D_x_prob, D_G_z1_prob, D_G_z2_prob, adr):
    
    plt.figure()
    x = [indi for indi in range(0, len(G_losses))]
    plt.plot(x, G_losses, 'g-')
    plt.plot(x, D_losses, 'b-')
    plt.legend(['Gen', 'Dsc']) 
    plt.savefig(os.path.join(adr, "plot_losses.jpg"))
    plt.close()
    
    plt.figure()
    x = [indi for indi in range(0, len(G_losses))]
    plt.plot(x, D_losses_fake, 'g-')
    plt.plot(x, D_losses_real, 'b-')
    plt.legend(['Dsc_real', 'Dsc_fake']) 
    plt.savefig(os.path.join(adr, "plot_losses2.jpg"))
    plt.close()
    
    plt.figure()
    x = [indi for indi in range(0, len(D_x_prob))]
    plt.plot(x, D_x_prob, 'g-')
    plt.plot(x, D_G_z1_prob, 'b-')
    plt.plot(x, D_G_z2_prob, 'r-')
    plt.legend(['D(R)', 'D(f)_1', 'D(f)_2']) 
    plt.savefig(os.path.join(adr, "plot_probs.jpg"))
    plt.close()



def animating_images(generated_fake, adr):
    
    img_list = []
    img_list.append(vutils.make_grid(generated_fake, padding=2, normalize=True))
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    ani.save(os.path.join(adr, "generated_images.gif")) 
    plt.close()
    

def change_scale(data):
    
    # Trimming first 
    low = np.percentile(data, 0.01)
    high = np.percentile(data, 99.99)
    data[data >= high] = high
    Min = 0
    Max = 255
    X_std = (data - data.min()) / (data.max() - data.min())
    X_scaled = X_std * (Max - Min) + Min
    return X_scaled 


def abs_images(data):
    data = np.absolute(data)
    return data

def plot_external_residual_generated_allimage2(data, adr, name, shifting_flag):
        
    row_no = 4
    col_no = 5
    _, axes = plt.subplots(row_no, col_no)
    
    for indi in range(0, row_no):
        for indj in range(0, col_no):
            image = data[indi * col_no + indj]
            if shifting_flag:
                image = abs_images(image)
            im = axes[indi][indj].imshow(image, cmap='gray')
            axes[indi][indj].axis('off')
            divider = make_axes_locatable(axes[indi][indj])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            ylabels = cax.get_yticks()
            cax.set_yticklabels([int(i) for i in ylabels], size=8, fontweight = 600)
            plt.colorbar(im, cax=cax)
            
    plt.savefig(os.path.join(os.path.join(adr, name + ".jpg")))
    plt.clf()
    plt.close()


def plot_all_images(input_images, generated_fake, generated_res_images, adr):
    
    input_images = np.squeeze(input_images.numpy())
    generated_fake = np.squeeze(generated_fake.numpy())
    generated_res_images = np.squeeze(generated_res_images.numpy())
    
    plot_external_residual_generated_allimage2(input_images, adr, "Input images", False)
    plot_external_residual_generated_allimage2(generated_fake, adr, "Generated_images", False)
    plot_external_residual_generated_allimage2(generated_res_images, adr, "Residuals", False)
    plot_external_residual_generated_allimage2(generated_res_images, adr, "Absolute residuals", True)
    
    
def plot_all_images_nifti(input_images, generated_fake, generated_res_images, adr):
    
    adr = os.path.join(adr, "Nifti_Seperate")
    if(os.path.exists(adr) == False):
        os.mkdir(adr)
        
    adr = os.path.join(adr, "Together")
    if(os.path.exists(adr) == False):
        os.mkdir(adr)
    img_affine = np.identity(4)
        
    for indi in range(0, input_images.shape[0]):
        concatenate = np.concatenate([input_images[indi, :, :], generated_res_images[indi, :, :]], axis = 0)
        concatenate = np.concatenate([concatenate, generated_fake[indi, :, :]], axis = 0) 
        
        img_adr = os.path.join(adr, "Image" + str(indi) + ".nii")
        array_img = nib.Nifti1Image(concatenate, img_affine)
        nib.save(array_img, img_adr)
        
    print("")
        

    
    
def plot_image2(data, adr, name):
    
    adr = os.path.join(adr, "Images")
    if(os.path.exists(adr) == False):
        os.mkdir(adr)

    _, axes = plt.subplots(len(data), 2)
    titles = ["Input image", "Residual", "Generated image"]
    for indi in range(0, len(data)):
        im = axes[indi][0].imshow(data[indi], cmap='gray')
        axes[indi][0].axis('off')
        axes[indi][0].title.set_text(titles[indi])
        divider = make_axes_locatable(axes[indi][0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ylabels = cax.get_yticks()
        cax.set_yticklabels([int(i) for i in ylabels], size=10, fontweight = 600)
        plt.colorbar(im, cax=cax)
        
    for indi in range(0, len(data)):
        if indi == 1:
            image = abs_images(data[indi])
            im = axes[indi][1].imshow(image, cmap='gray')
            axes[indi][1].axis('off')
            axes[indi][1].title.set_text("Absolute Residual")
            divider = make_axes_locatable(axes[indi][1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            ylabels = cax.get_yticks()
            cax.set_yticklabels([int(i) for i in ylabels], size=10, fontweight = 600)
            plt.colorbar(im, cax=cax)
        else: 
            axes[indi][1].axis('off')
            
        
    plt.savefig(os.path.join(os.path.join(adr, name + ".jpg")))
    plt.clf()
    plt.close()
    
    
def plot_external_residual_generated_image2(input_images, generated_fake, generated_res_images, adr):
    
    input_images = np.squeeze(input_images.numpy())
    generated_res_images = np.squeeze(generated_res_images.numpy())
    generated_fake = np.squeeze(generated_fake.numpy())
    
    for indi in range(0, input_images.shape[0]):
        plot_image2([input_images[indi, :, :], generated_res_images[indi, :, :], generated_fake[indi, :, :]],  \
                   adr, "images_" + str(indi) + ".jpeg")

        
        
def plot_image_nifti(data, adr, name, Image_types, img_affine):
    
    for indi in range(0, len(Image_types)):
        img_adr = os.path.join(adr, Image_types[indi], name)
        array_img = nib.Nifti1Image(data[indi], img_affine)
        nib.save(array_img, img_adr)


        
def plot_external_residual_generated_image_nifti(input_images, generated_fake, generated_res_images, adr):
    
    adr = os.path.join(adr, "Nifti_Seperate")
    Image_types = ["Input", "Residuals", "Generated"]
    if(os.path.exists(adr) == False):
        os.mkdir(adr)
        os.mkdir(os.path.join(adr, "Input"))
        os.mkdir(os.path.join(adr, "Generated"))
        os.mkdir(os.path.join(adr, "Residuals"))
    
    input_images = np.squeeze(input_images.numpy())
    generated_res_images = np.squeeze(generated_res_images.numpy())
    generated_fake = np.squeeze(generated_fake.numpy())
    
    img_affine = np.identity(4)
    
    for indi in range(0, input_images.shape[0]):
        plot_image_nifti([input_images[indi, :, :], generated_res_images[indi, :, :], generated_fake[indi, :, :]],  \
                   adr, "images_" + str(indi) + ".nii", Image_types, img_affine)
        
        
def generate_masked_images(images, indices, mask):
    
    for indi in range(0, images.shape[0]):
        images[indi, 0, :, :] = torch.mul(images[indi, 0, :, :], mask[:, :, indices[indi]])

    return images

        
        
'''          
def Plot_images_main(generated_fake, adr, name, scaling_flag, shifting_flag):
    
    images_data = generated_fake.numpy()
    images_data = np.squeeze(images_data)
    images = []
    for indi in range(0, images_data.shape[0]):
        if (scaling_flag):
            img = change_scale(images_data[indi, :, :])
        elif (shifting_flag):
            img = abs_images(images_data[indi, :, :])
        else:
            img = images_data[indi, :, :]
        images.append(img)
        
    row_no = 4
    col_no = 5
    all_concatenated_images = np.zeros((1, col_no * images[0].shape[1]))
    
    for indi in range(0, row_no):
        for indj in range(0, col_no):
            if (indj == 0):
                concatenated_images = images[indi * col_no]
            else:
                concatenated_images = np.concatenate((concatenated_images, images[indi * col_no + indj]), axis=1)
                 
            if ((indi == 0) and (indj == col_no - 1)):
                all_concatenated_images = concatenated_images
            elif ((indi > 0) and (indj == col_no - 1)):
                all_concatenated_images = np.concatenate((all_concatenated_images, concatenated_images), axis=0) 
        
    plot_image(all_concatenated_images, adr, name)
    
    
def Plot_images_main2(generated_fake, adr, name, scaling_flag, shifting_flag):
    
    images_data = generated_fake.numpy()
    images = np.squeeze(images_data)
        
    row_no = 4
    col_no = 5
    all_concatenated_images = np.zeros((1, col_no * images[0].shape[1]))
    
    for indi in range(0, row_no):
        for indj in range(0, col_no):
            if (indj == 0):
                concatenated_images = images[indi * col_no]
            else:
                concatenated_images = np.concatenate((concatenated_images, images[indi * col_no + indj]), axis=1)
            if ((indi == 0) and (indj == col_no - 1)):
                all_concatenated_images = concatenated_images
            elif ((indi > 0) and (indj == col_no - 1)):
                all_concatenated_images = np.concatenate((all_concatenated_images, concatenated_images), axis=0) 

    if (scaling_flag):
        img = change_scale(all_concatenated_images)
    elif (shifting_flag):
        img = abs_images(all_concatenated_images)
    else:
        img = all_concatenated_images
        
    plot_image(img, adr, name)
    
    
    
def plot_image(data, adr, name):
    
    # plot image
    ax = plt.subplot()
    im = ax.imshow(data, cmap='gray')
    plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ylabels = cax.get_yticks()
    cax.set_yticklabels([int(i) for i in ylabels], size=12, fontweight = 700)
    plt.colorbar(im, cax=cax)
    
    plt.savefig(os.path.join(os.path.join(adr, name + ".jpg")))
    plt.clf()
    ax.clear()
    plt.close()


    
def plot_generated_images(generated_fake, generated_res_images, adr):  

    Plot_images_main2(generated_fake, adr, "scaled2_generated_images", True, False)
    Plot_images_main(generated_fake, adr, "generated_images", False, False) # scaling_flag, shifting_flag
    Plot_images_main(generated_fake, adr, "scaled_generated_images", True, False)
    Plot_images_main(generated_res_images, adr, "residual_images", False, False)
    Plot_images_main(generated_res_images, adr, "abs_residual_images", False, True)

    
        
def plot_external_residual_generated_image(input_images, generated_fake, generated_res_images, adr):
    
    input_images = np.squeeze(input_images.numpy())
    generated_res_images = np.squeeze(generated_res_images.numpy())
    generated_fake = np.squeeze(generated_fake.numpy())
    
    for indi in range(0, input_images.shape[0]):
        # concatenate data
        concatenated_data = np.concatenate((input_images[indi, :, :], generated_res_images[indi, :, :]), axis = 0)
        concatenated_data = np.concatenate((concatenated_data, generated_fake[indi, :, :]), axis = 0)
        
        # scale data 
        concatenated_data = change_scale(concatenated_data)
        plot_image(concatenated_data, adr, "images_" + str(indi) + ".jpeg")

    
'''    

