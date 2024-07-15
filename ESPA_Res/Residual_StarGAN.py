'''
The original code is from
From https://github.com/eriklindernoren/PyTorch-GAN#stargan
'''

import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch
import pandas as pd
import nibabel as nib
import random
from torch.utils import data
import torchvision.transforms as transforms
import torch.nn.parallel

gpu_number = "0, 1, 2, 3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number

def configuration_for_reproducibility(manual_seed):
    random.seed(manual_seed) # generating random numbers in Python
    torch.manual_seed(manual_seed)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

################## make directories #########################
imges_dir = os.path.join("./save" + str(gpu_number), "images")
model_dir = os.path.join("./save" + str(gpu_number), "saved_models")
results_dir = os.path.join("./save" + str(gpu_number), "Results")
os.makedirs(imges_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
manual_seed = 1234
configuration_for_reproducibility(manual_seed)
Start = time.time()
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

###########################################################################################
#                                         RESNET_Generator
###########################################################################################
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [
            nn.Conv2d(in_features, in_features, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(in_features, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(in_features, affine=True, track_running_stats=True),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, img_shape=(1, 192, 192), res_blocks=9, c_dim=5):
        super(GeneratorResNet, self).__init__()
        channels, img_size, _ = img_shape

        # Initial convolution block
        # 1 in channels + c_dim + 1 is for noise
        model = [
            nn.Conv2d(channels + c_dim + 1, 64, 7, stride=1, padding=3, bias=False), 
            nn.InstanceNorm2d(64, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        curr_dim = 64
        for _ in range(2):
            model += [
                nn.Conv2d(curr_dim, curr_dim * 2, 4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ]
            curr_dim *= 2

        # Residual blocks
        for _ in range(res_blocks):
            model += [ResidualBlock(curr_dim)]

        # Upsampling
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(curr_dim, curr_dim // 2, 4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ]
            curr_dim = curr_dim // 2

        # Output layer
        model += [nn.Conv2d(curr_dim, channels, 7, stride=1, padding=3)]

        self.model = nn.Sequential(*model)

    def forward(self, x, c, noise):
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        img = torch.cat((x, c, noise), 1)
        residual = self.model(img) + x
        return residual, torch.tanh(residual)

###########################################################################################
#                                         Discriminator
###########################################################################################

class Discriminator(nn.Module):
    def __init__(self, img_shape=(3, 128, 128), c_dim=5, n_strided=6):
        super(Discriminator, self).__init__()
        channels, img_size, _ = img_shape

        def discriminator_block(in_filters, out_filters):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1), nn.LeakyReLU(0.01)]
            return layers

        layers = discriminator_block(channels, 64)
        curr_dim = 64
        for _ in range(n_strided - 1):
            layers.extend(discriminator_block(curr_dim, curr_dim * 2))
            curr_dim *= 2

        self.model = nn.Sequential(*layers)

        # Output 1: PatchGAN
        self.out1 = nn.Conv2d(curr_dim, 1, 3, padding=1, bias=False)
        # Output 2: Class prediction
        kernel_size = img_size // 2 ** n_strided
        self.out2 = nn.Conv2d(curr_dim, c_dim, kernel_size, bias=False)

    def forward(self, img):
        feature_repr = self.model(img)
        out_adv = self.out1(feature_repr)
        out_cls = self.out2(feature_repr)
        return out_adv, out_cls.view(out_cls.size(0), -1)
    
#############################################################################################
#                                  Util 
#############################################################################################
def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.tensor(np.random.random((real_samples.size(0), 1, 1, 1)), dtype = torch.float).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates, _ = D(interpolates)
    fake = torch.tensor(np.ones(d_interpolates.shape), requires_grad=False, dtype = torch.float).to(device)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def sample_images(batches_done, val_dataloader, c_dim, Target_labels, generator, opt):
    """Saves a generated sample of domain translations"""
    
    val_imgs, _ = next(iter(val_dataloader))
    val_imgs = val_imgs.to(device)
    Target_labels = Target_labels.to(device)
    img_samples = None
    
    for i in range(10):
        img, label = val_imgs[i], Target_labels
        # giving similar noise to all target scanners
        fixed_noise = make_noise_mat(1, img.shape[1], img.shape[2], opt.nz).to(device)
        fixed_noise = fixed_noise.repeat(len(label), 1, 1, 1)
        # Repeat for number of label changes
        imgs = img.repeat(len(label), 1, 1, 1)
        # Generate translations
        res, gen_imgs = generator(imgs, label, fixed_noise)
        # Concatenate images by width
        gen_imgs = torch.cat([x for x in gen_imgs.data], -1)
        img_sample = torch.cat((img.data, gen_imgs), -1)
        # Add as row to generated samples
        img_samples = img_sample if img_samples is None else torch.cat((img_samples, img_sample), -2)

    save_image(img_samples.view(1, *img_samples.shape), os.path.join(imges_dir, "images" + str(batches_done) + ".png"), normalize=True)


def check_tensor_equality(model1, model2):
    flag = True
    for a, b in zip(model1.values(), model2.values()):
        if torch.all(a.eq(b)):
            pass
        else:
            flag = False
    if (flag):
        print("Equal!")
    else:
        print("Unequal!")
        

def losses_on_validationset(batches_done, val_dataloader, c_dim, Target_labels, generator, discriminator):
    
    losses_D = []
    losses_G = []
    
    for batch, (imgs, labels) in enumerate(val_dataloader):
        
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        sampled_c = torch.tensor(generating_fake_labels(imgs.shape[0], c_dim), dtype = torch.float).to(device)
        fixed_noise = make_noise_mat(imgs.shape[0], imgs.shape[2], imgs.shape[3], opt.nz).to(device)
        res, fake_imgs = generator(imgs, sampled_c, fixed_noise)
        real_validity, pred_cls = discriminator(imgs)
        fake_validity, _ = discriminator(fake_imgs.detach())
        gradient_penalty = compute_gradient_penalty(discriminator, imgs.data, fake_imgs.data)
        RV = torch.mean(real_validity)
        FV = torch.mean(fake_validity) 
        loss_D_cls = criterion_cls(pred_cls, labels)
        losses_D.append([RV.item(), FV.item(), gradient_penalty.item(), loss_D_cls.item()])
        
        # generator 
        res, recov_imgs = generator(fake_imgs, labels, fixed_noise)
        _, pred_cls = discriminator(fake_imgs)
        loss_G_cls = criterion_cls(pred_cls, sampled_c)
        loss_G_rec = criterion_cycle(recov_imgs, imgs)
        losses_G.append([0.0, loss_G_cls.item(), loss_G_rec.item()]) 

    losses_G = np.mean(np.array(losses_G), axis=0) 
    losses_D = np.mean(np.array(losses_D), axis=0) 
    return losses_G, losses_D

    
    
    
def read_data_target_scanner(directory, scanner_name, CV_directory, CV_no, file_type):
    
    # Read excel file that contains the name of images
    if CV_no == None: 
        excel_file_adr = os.path.join(CV_directory, "Names_" + file_type + ".xlsx")
    else:
        excel_file_adr = os.path.join(CV_directory, "Names_" + file_type + "_fold" + str(CV_no) + ".xlsx")
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


def read_data_external_scanner(directory, scanner_name, file_type):
    
    # Read excel file that contains the name of images
    excel_file_adr = os.path.join(directory, scanner_name, "External_scanner_image_info_" + file_type + ".xlsx")
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


def making_labels(scanner_name):
    if (scanner_name == "OASIS"):
        return [1., 0., 0., 0., 0.]
    elif (scanner_name == "ge"):
        return [0., 1., 0., 0., 0.]
    elif (scanner_name == "philips"):
        return [0., 0., 1., 0., 0.]
    elif (scanner_name == "prisma"):
        return [0., 0., 0., 1., 0.]
    elif (scanner_name == "trio"):
        return [0., 0., 0., 0., 1.]
    
def duplicate_list(lst, dup_time):
    dp_lst = []
    for indi in range(0, dup_time):
        dp_lst.append(lst)
    return dp_lst 


def read_datasets(opt, target_scanners, file_type):
    
    external_scanner_name = "scanner1"
    target_scanner_labels = []
    
    target_scanner_name = target_scanners[0]
    Target_scanner_data = read_data_target_scanner(opt.target_scanner_image_adrs,\
                              target_scanner_name, opt.CVfolds_adrs, opt.CV_no, file_type)
    target_scanner_labels = target_scanner_labels + duplicate_list(making_labels(target_scanners[0]), Target_scanner_data.shape[2])
    
    for indi in range(1, len(target_scanners)):
        target_scanner_name = target_scanners[indi]
        target_scanner_images = read_data_target_scanner(opt.target_scanner_image_adrs,\
                              target_scanner_name, opt.CVfolds_adrs, opt.CV_no, file_type)
        Target_scanner_data = np.concatenate((Target_scanner_data, target_scanner_images), axis = 2)
        target_scanner_labels = target_scanner_labels + duplicate_list(making_labels(target_scanners[indi]), target_scanner_images.shape[2])
        
    target_scanner_labels = np.array(target_scanner_labels)
    External_scanner_images = read_data_external_scanner(opt.external_scanner_image_adrs, \
                                                         external_scanner_name, file_type)
    External_scanner_labels = np.array(duplicate_list(making_labels("OASIS"), External_scanner_images.shape[2]))
    
    return Target_scanner_data, target_scanner_labels, External_scanner_images, External_scanner_labels


def set_loaders_seperately(Target_scanner_images, target_scanner_labels, \
            External_scanner_images, External_scanner_labels, opt):
    
    # Loading the dataset. 
    Target_scanner_dataset = ScannerDataset(Target_scanner_images, target_scanner_labels)
    External_scanner_dataset = ScannerDataset(External_scanner_images, External_scanner_labels)
    
    Target_scanner_loader = torch.utils.data.DataLoader(
        Target_scanner_dataset, batch_size = opt.batch_size, shuffle = True,
        num_workers = opt.num_workers, pin_memory = True, sampler = None)
    
    External_scanner_loader = torch.utils.data.DataLoader(
        External_scanner_dataset , batch_size = opt.batch_size, shuffle = True,
        num_workers = opt.num_workers, pin_memory = True, sampler = None)
    
    return  Target_scanner_loader, External_scanner_loader


def set_loader(Target_scanner_images, target_scanner_labels, \
            External_scanner_images, External_scanner_labels, opt, shuffle_flag):
    
    if (Target_scanner_images is None):
        images = External_scanner_images
        lables = External_scanner_labels

    else:
        # concatenate data
        images = np.concatenate((Target_scanner_images, External_scanner_images), axis = 2)
        lables = np.concatenate((target_scanner_labels, External_scanner_labels), axis = 0)
        
    # define dataset
    img_dataset = ScannerDataset(images, lables)
    
    # define dataloader
    if (shuffle_flag):
        img_loader = torch.utils.data.DataLoader(\
            img_dataset, batch_size = opt.batch_size, shuffle = True,\
            num_workers = opt.num_workers, pin_memory = True, sampler = None)
    else:
        img_loader = torch.utils.data.DataLoader(\
            img_dataset, batch_size = opt.batch_size, shuffle = False,\
            num_workers = opt.num_workers, pin_memory = False, sampler = None)

    return  img_loader


def generating_fake_labels_for_design3(label_no, scanner_no, scanner_range):
    
    lst = []
    for indi in range(0, label_no):
        if (len(scanner_range) == 1):
            label = [1] + [0] * (scanner_no - 1)
            
        elif (len(scanner_range) > 1):
            indx = random.randint(scanner_range[0], scanner_range[1])
            label = np.zeros((scanner_no))
            label[indx] = 1 
        lst.append(label)
    return torch.FloatTensor(lst)

def make_noise_mat(batch_size, image_x, image_y, nz):
    
    noise = torch.randn(batch_size, nz, dtype = torch.float)
    concat_matrix = torch.randn(1, image_x, image_y)
    for indi in range(0, batch_size):
        image_size = image_x * image_y
        image_mat_size = math.ceil(image_size/nz)
    
        repeated_noise = noise[indi, :].repeat(image_mat_size)
        repeated_noise = torch.reshape(repeated_noise[0:image_size], (image_x, image_y))

        repeated_noise = repeated_noise[None, :, :]
        concat_matrix = torch.cat((concat_matrix, repeated_noise), 0)

    return concat_matrix[1:, None, :, :]

def generating_fake_labels(label_no, scanner_no):
    lst = []
    for indi in range(0, label_no):
        indx = random.randint(0, scanner_no - 1)
        label = np.zeros((scanner_no))
        label[indx] = 1 
        lst.append(label)
    return np.array(lst)
    
###################################################################################################################
#                                               ScannerDataset
###################################################################################################################


class ScannerDataset(data.Dataset):
    def __init__(self,
                 images, # numpy array of stacked axial slices
                 labels
                ):
        self.dataset = images
        self.labels = labels

    def __len__(self):
        return self.dataset.shape[2]

    def __getitem__(self, index: int):
        
        axial_slice = torch.tensor(self.dataset[:, :, index], dtype = torch.float)
        axial_slice = axial_slice[None, :, :]
        axial_slice = self.resize_images(axial_slice)
        axial_slice = self.normalizing_images(axial_slice)
        label = torch.tensor(self.labels[index, :], dtype = torch.float)
        return axial_slice, label 
    
    def normalizing_images(self, slice):
        return  (2 * ((slice - slice.min())/(0.0001 + slice.max() - slice.min()))) - 1

    def resize_images(self, slice):
        size_x = 192
        size_y = 192 
        desired_size_x = 152
        desired_size_y = 188 
        
        resized_slice = torch.zeros([1, size_x, size_y], dtype = torch.float)
        x_changes = (int) ((size_x - desired_size_x)/2)
        y_changes = (int) ((size_y - desired_size_y)/2)
        resized_slice[:, x_changes: x_changes + desired_size_x, y_changes: y_changes + desired_size_y] = slice
        return resized_slice
    

###################################################################################################################
#                                               Code
###################################################################################################################
################## read arguments #########################
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--target_scanner_image_adrs", type=str, default="./Dataset/TargetScanners", help="directory of multi-scanner data (target scanners)")
parser.add_argument("--external_scanner_image_adrs", type=str, default= "./Dataset/ExternalScanner", help="directory of source data (source scanners)")
parser.add_argument("--CVfolds_adrs", type=str, default="./Dataset/CV_Folds", help="directory to images of CV fold")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr_Gen", type=float, default=0.0002, help="adam: learning rate of generator")
parser.add_argument("--lr_Dsc", type=float, default=0.0002, help="adam: learning rate of discriminator")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default= 224, help="size of image height")
parser.add_argument("--img_width", type=int, default= 224, help="size of image width")
parser.add_argument("--channels", type=int, default= 1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default= 100, help="interval between saving generator samples")
parser.add_argument("--checkpoint_interval", type=int, default= 2, help="interval between model checkpoints")
parser.add_argument("--validation_loss_interval", type=int, default= 300, help="interval for calculating losses on validation")
parser.add_argument("--residual_blocks", type=int, default= 9, help="number of residual blocks in generator")
parser.add_argument("--lambda_cls", type=float, default= 1.0, help="coefficient for auxilary loss")
parser.add_argument("--lambda_rec", type=float, default= 10.0, help="coefficient for reconstruction loss")
parser.add_argument("--lambda_gp", type=float, default= 10.0, help="coefficient for gradient penalty loss")
parser.add_argument("--CV_no", type=int, default= 0, help="CV fold for UCDavis")
parser.add_argument("--num_workers", type=int, default= 0, help="Number of workers")
parser.add_argument("--nz", type=int, default=192, help="noise dimention")
parser.add_argument(
    "--selected_attrs",
    "--list",
    nargs="+",
    help="scanner labels",
    default=["OASIS", "GE", "Philips", "Prisma", "Trio"],
)
parser.add_argument("--n_critic", type=int, default=5, help="number of training iterations for WGAN discriminator")
opt = parser.parse_args()


# Loss weights
target_scanners = ["ge", "philips", "prisma", "trio"]
lambda_cls = opt.lambda_cls
lambda_rec = opt.lambda_rec
lambda_gp = opt.lambda_gp

c_dim = len(opt.selected_attrs)
img_shape = (opt.channels, opt.img_height, opt.img_width)

Target_labels = [[0., 1., 0., 0., 0.], 
                 [0., 0., 1., 0., 0.], 
                 [0., 0., 0., 1., 0.],
                 [0., 0., 0., 0., 1.]]
Target_labels = torch.tensor(Target_labels, dtype = torch.float)


############################# defining losses #############################
# Loss functions
criterion_cycle = torch.nn.L1Loss()

def criterion_cls(logit, target):
    return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)


############################ Initialize generator and discriminator ############################
generator = GeneratorResNet(img_shape=img_shape, res_blocks=opt.residual_blocks, c_dim=c_dim).to(device)
discriminator = Discriminator(img_shape=img_shape, c_dim=c_dim).to(device)

if (device.type == "cuda"):
    generator = nn.DataParallel(generator, [0, 1, 2, 3])
    discriminator = nn.DataParallel(discriminator, [0, 1, 2, 3])

criterion_cycle.to(device)

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load("saved_models/generator_%d.pth" % opt.epoch))
    discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth" % opt.epoch))
else:
    configuration_for_reproducibility(manual_seed)
    generator.apply(weights_init_normal)
    configuration_for_reproducibility(manual_seed)
    discriminator.apply(weights_init_normal)
    
############################## Optimizers ##############################
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr_Gen, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr_Dsc, betas=(opt.b1, opt.b2))


##################################### Configure dataloaders #####################################
# read data and labels for train
shuffle_flag = True
Target_scanner_images, target_scanner_labels, External_scanner_images, External_scanner_labels\
 = read_datasets(opt, target_scanners, "train")
 
dataloader = set_loader(Target_scanner_images, target_scanner_labels, \
            External_scanner_images, External_scanner_labels, opt, shuffle_flag)

# ------------------------------------------------------------------------------
#                                  Training
# ------------------------------------------------------------------------------
losses_D = [] 
losses_G = [] 
losses_D_val = [] 
losses_G_val = [] 
saved_samples = []
start_time = time.time()
iters = 0

for epoch in range(opt.epoch, opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        # Model inputs
        imgs = imgs.to(device)
        labels = labels.to(device)

        # Sample labels as generator inputs. 
        sampled_c = torch.tensor(generating_fake_labels(imgs.shape[0], c_dim), dtype = torch.float).to(device)
        # Generate fake batch of images

        fixed_noise = make_noise_mat(imgs.shape[0], imgs.shape[2], imgs.shape[3], opt.nz).to(device)
        res, fake_imgs = generator(imgs, sampled_c, fixed_noise)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Real images
        
        real_validity, pred_cls = discriminator(imgs) 
        # Fake images
        fake_validity, _ = discriminator(fake_imgs.detach())

        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, imgs.data, fake_imgs.data)
        # Adversarial loss
        RV = torch.mean(real_validity)
        FV = torch.mean(fake_validity)
        loss_D_adv = -RV + FV + lambda_gp * gradient_penalty
        # Classification loss
        loss_D_cls = criterion_cls(pred_cls, labels) # pred_cls: classifier prob for real images, label of real images
        # Total loss
        loss_D = loss_D_adv + lambda_cls * loss_D_cls

        loss_D.backward()
        optimizer_D.step()
        losses_D.append([RV.item(), FV.item(), gradient_penalty.item(), loss_D_cls.item()])
        optimizer_G.zero_grad()
        
        # Every n_critic times update generator
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            # Translate and reconstruct image
            generator.train()
            res, gen_imgs = generator(imgs, sampled_c, fixed_noise)
            res, recov_imgs = generator(gen_imgs, labels, fixed_noise)
            # Discriminator evaluates translated image
            fake_validity, pred_cls = discriminator(gen_imgs)
            # Adversarial loss
            FV_G = torch.mean(fake_validity)
            loss_G_adv = -FV_G
            # Classification loss
            loss_G_cls = criterion_cls(pred_cls, sampled_c)
            # Reconstruction loss
            loss_G_rec = criterion_cycle(recov_imgs, imgs)
            # Total loss
            loss_G = loss_G_adv + lambda_cls * loss_G_cls + lambda_rec * loss_G_rec
            losses_G.append([FV_G.item(), loss_G_cls.item(), loss_G_rec.item()]) # change
            loss_G.backward()
            optimizer_G.step()

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - start_time) / (batches_done + 1))

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D adv: %f, aux: %f] [G loss: %f, adv: %f, aux: %f, cycle: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_D_adv.item(),
                    loss_D_cls.item(),
                    loss_G.item(),
                    loss_G_adv.item(),
                    loss_G_cls.item(),
                    loss_G_rec.item(),
                    time_left,
                )
            )

            
        iters +=1



    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), os.path.join(model_dir, "generator_" + str(epoch) + ".pth"))
        torch.save(discriminator.state_dict(), os.path.join(model_dir, "discriminator_" + str(epoch) + ".pth"))
        # Rewite losses
        cols_D = ["RV_D", "FV_D", "grsd_prnslty", "cls_R"]
        cols_G = ["FV_G", "cls_F", "Recon_loss"]
        df_D = pd.DataFrame(losses_D, columns = cols_D)
        df_G = pd.DataFrame(losses_G, columns = cols_G)
        df_D.to_excel(os.path.join(results_dir, "losses_D_train.xlsx"), index = False)
        df_G.to_excel(os.path.join(results_dir, "losses_G_train.xlsx"), index = False)



    
end = time.time()
print("Start:" + str(time.asctime(time.localtime(Start))))
print("End:" + str(time.asctime( time.localtime(end))))
        
        
        
        
