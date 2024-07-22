from __future__ import print_function

import os
import sys
import argparse
import time
import math

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
from DataSet2 import HarmonizationDataSet

from util import AverageMeter
from util import set_optimizer, save_model
from networks.UNET import UnSupConMISPEL
from losses import MISPEL_losses
import random
import numpy as np
import pandas as pd
import nibabel as nib
from util import write_into_excel_file, make_excel_file, calculate_performance
from os.path import exists


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

def change_adrs(adrs, substring, string):
    for i in range(0, adrs.shape[0]):
        for j in range(0, adrs.shape[1]):
            adrs[i][j] = adrs[i][j].replace(substring, string)
    return adrs


def parse_option(gpu_number):
      
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--calculate_performance_Flag', type=bool, default=False,
                        help='calculate performances')
    
    parser.add_argument('--CV_no', type=int, default=1,
                        help='Cross validation no')
                        
    parser.add_argument('--performance_Freq', type=int, default=2,
                        help='frequency of calculating performance measures')
                        
    parser.add_argument('--Max_index_of_train_datasets', type=int, default=30,
                        help='frequency of calculating performance measures')
                        
    parser.add_argument('--data_frequency_step1', type=int, default=5,
                        help='frequency of seeing new set of data') 
                        
    parser.add_argument('--data_frequency_step2', type=int, default=2,
                        help='frequency of seeing new set of data') 

    parser.add_argument('--print_freq', type=int, default=2,
                        help='print frequency')
    
    parser.add_argument('--no_latent_embeddings', type=int, default=6,
                        help='number of components in th elatent embedding (L in paper).')
    
    parser.add_argument('--head', type=str, default="linear_decoder",
                        help='Type of head network which is the decoder in MISPEL.')
    
    parser.add_argument('--train_data_adr', type=str, default="./Dataset/Data_For_Loader_trainvalidation/saved_data/training_images_one_epoch_nifti_0.xlsx",
                        help='Address to augmented train images.')
                        
    parser.add_argument('--validation_data_adr', type=str, default="./Dataset/Data_For_Loader_trainvalidation/saved_data/validation_images_one_epoch_nifti_0.xlsx",
                        help='Address to augmented validation images.')
    
    parser.add_argument('--mask_adr', type=str, default = "Dataset/cropped_JHU_MNI_SS_T1_Brain_Mask.nii.gz",
                        help='print frequency')
    
    parser.add_argument('--output_excel_name_step1_train', type=str, default = "",
                        help='name of excel file for step 1 train losses')
    
    parser.add_argument('--output_excel_name_step1_validation', type=str, default = "",
                        help='name of excel file for step 1 validation losses')
    
    parser.add_argument('--output_excel_name_step2_train', type=str, default = "",
                        help='name of excel file for step 2 train losses')
    
    parser.add_argument('--output_excel_name_step2_validatio', type=str, default = "",
                        help='name of excel file for step 2 validation losses')
      
    parser.add_argument('--padding_axial_dim', type=int, default=2,
                        help='print frequency')
    
    parser.add_argument('--no_axial_slices', type=int, default=152,
                        help='print frequency')
    
    parser.add_argument('--image_input_info', type=str, default="./Dataset/one_selected_scanner_data.xlsx",
                        help='this excel file should have the info of the images')

    parser.add_argument('--batch_size', type=int, default= 28,
                        help='batch_size')
    
    parser.add_argument('--num_workers', type=int, default= 0,
                        help='num of workers to use')
    
    parser.add_argument('--scanner_no', type=int, default= 4,
                        help='num of scanners')
    
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')
    
    parser.add_argument('--epochs_step1', type=int, default=2,
                        help='number of training epochs in step1')
    
    parser.add_argument('--epochs_step2', type=int, default=2,
                        help='number of training epochs in step2')
    
    parser.add_argument('--save_freq_step1', type=int, default=50,
                        help='frequency for saving model for step1')
    
    parser.add_argument('--save_freq_step2', type=int, default=100,
                        help='frequency for saving model for step2')
    
    parser.add_argument('--train_valid_split', type=float, default=0.8,
                        help='train validation split ratio')
    
    parser.add_argument('--lambda1', type=float, default = 0.3,
                        help='Step1_Lrecon_coefficient')
    parser.add_argument('--lambda2', type=float, default = 1.0,
                        help='Step1_Lcoupling_coefficient')
    parser.add_argument('--lambda3', type=float, default = 1.0,
                        help='Step2_Lrecon_coefficient')
    parser.add_argument('--lambda4', type=float, default = 4.0,
                        help='Step2_Lharm_coefficient')
    
    parser.add_argument('--learning_rate_step1', type=float, default=0.0001,
                        help='learning rate')
                        
    parser.add_argument('--learning_rate_step2', type=float, default=0.001,
                        help='learning rate')
                        
    parser.add_argument('--eps', type=float, default=0.00000001,
                        help='learning rate')                      
      
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='UNET')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'path'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')
    
    # method
    parser.add_argument('--method', type=str, default='SimCLR',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.5,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    
    opt = parser.parse_args()
    
    ############ Added by Mahbaneh  ################
    opt.cosine = True
    opt.syncBN = False
    ############ Added by Mahbaneh  ################
    
    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './save' + gpu_number + '/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save' + gpu_number + '/SupCon/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate_step1,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate_step1 * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate_step1 - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate_step1

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    # ignoring name of model 
    opt.model_name = "model"

    opt.save_folder = os.path.join(opt.model_path, opt.model_name, "training")
    opt.save_folder1 = os.path.join(opt.model_path, opt.model_name, "training_step1")
    opt.save_folder2 = os.path.join(opt.model_path, opt.model_name, "training_step2")
    
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
        os.makedirs(opt.save_folder1)
        os.makedirs(opt.save_folder2)
        
    # Excel files for epochs
    opt.output_excel_name_step1_train = os.path.join(opt.save_folder1, "train_epoches.xlsx")
    opt.output_excel_name_step1_validation = os.path.join(opt.save_folder1, "validation_epoches.xlsx")
    opt.output_excel_name_step2_train = os.path.join(opt.save_folder2, "train_epoches.xlsx")
    opt.output_excel_name_step2_validation = os.path.join(opt.save_folder2, "validation_epoches.xlsx")
    filenames = [opt.output_excel_name_step1_train, opt.output_excel_name_step1_validation,
                 opt.output_excel_name_step2_train, opt.output_excel_name_step2_validation]
                 
    if os.path.exists(os.path.join("save" + str(gpu_number), "Results")) == False:
        os.mkdir(os.path.join("save" + str(gpu_number), "Results"))
    
    for filename in filenames:
        if (exists(filename) == False): 
            make_excel_file(filename)
            write_into_excel_file(filename, ["Epochs", "Loss", "Loss1", "Loss2"])
    return opt
    
def Read_data_info(opt):
    
    image_info = pd.read_excel(opt.image_input_info)
    
    # Generate the addresses 
    image_adr_list = []
    gmm_adr_list = []
    adr_scanner_id = []
    adr_slice_id = []
    
    for indi in range(0, image_info.shape[0]):
        
        # Prepare address of the slice
        image_adr = os.path.join("Dataset/FinalFiles", image_info["Folder_name"][indi]
                                     , image_info["SBJ_name"][indi], image_info["Nifti_fileName"][indi])
        image_adr_list.extend([image_adr] * opt.no_axial_slices) 
        
        # Prepare address of the GMM
        gmm_adr = os.path.join("Dataset/FinalFiles", image_info["GGM_Folder_name"][indi]
                                     , image_info["SBJ_name"][indi], image_info["GGM_File_name"][indi] + ".sav")
        gmm_adr_list.extend([gmm_adr] * opt.no_axial_slices) 
        
        adr_scanner_id.extend([image_info["ScannerID"][indi]] * opt.no_axial_slices)
        adr_slice_id.extend(list(range(2, opt.no_axial_slices + 2)))
        
    all_list = list(zip(image_adr_list, adr_scanner_id, adr_slice_id, gmm_adr_list))
        
    return all_list


def set_model(opt):

    model = UnSupConMISPEL(name = opt.model, head = opt.head, 
                           latent_embedding_no = opt.no_latent_embeddings, scanner_no = opt.scanner_no)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        cudnn.benchmark = True

    return model


def set_losses(opt):

    # criterion step1
    criterion1 = MISPEL_losses(1, opt.lambda1, opt.lambda2, opt.lambda3, opt.lambda4, opt.batch_size, opt.scanner_no)
    
    # criterion step2
    criterion2 = MISPEL_losses(2, opt.lambda1, opt.lambda2, opt.lambda3, opt.lambda4, opt.batch_size, opt.scanner_no)

    if torch.cuda.is_available():
        criterion1 = criterion1.cuda()
        criterion2 = criterion2.cuda()

    return criterion1, criterion2


def validation(valid_loader, model, criterion, optimizer, epoch, opt, 
               tarining_step_number, mask, validation_data, performance_flag):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    validation_loss = AverageMeter()
    end = time.time()
    validation_loss1 = AverageMeter()
    validation_loss2 = AverageMeter()
    
    n = (int)(((opt.scanner_no)*(opt.scanner_no - 1))/2)
    SSIM_sum = [0 for i in range(0, n)]
    MAE_sum = [0 for i in range(0, n)]
    image_count = 0
    
    model.eval()
    with torch.no_grad():
        for idx, (images, index) in enumerate(valid_loader):
            
            image_count += len(index)
            image_no = images[0].shape[0]
            data_time.update_other_metrics(time.time() - end)
            
        
            if torch.cuda.is_available():
                for i in range(0, len(images)):
                    images[i] = images[i].cuda(non_blocking=True)
                    
            embeddings, reconstructed_images = model(images, tarining_step_number) # tensor(btach-size * aug, features_dim)
            reconstructed_images = [torch.unsqueeze(reconstructed_image, dim = 1) for reconstructed_image in reconstructed_images]

            loss, loss1, loss2 = criterion(images, embeddings, reconstructed_images)            
            validation_loss.update_other_metrics(loss.item(), image_no)
            validation_loss1.update_other_metrics(loss1.item(), image_no)
            validation_loss2.update_other_metrics(loss2.item(), image_no)
            
            # measure elapsed time
            batch_time.update_other_metrics(time.time() - end)
            end = time.time()
            
            if (idx + 1) % opt.print_freq == 0:
                print('Validation: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(valid_loader), batch_time=batch_time,
                   data_time=data_time, loss=validation_loss))
                sys.stdout.flush()
     
    return validation_loss.avg, validation_loss1.avg, validation_loss2.avg, SSIM_sum, MAE_sum


def train(train_loader, model, criterion, optimizer, epoch, opt, tarining_step_number,
           mask, train_data, performance_flag):
    
    """one epoch training"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    train_loss = AverageMeter()
    train_loss1 = AverageMeter()
    train_loss2 = AverageMeter()
  
    model.train()
    torch.backends.cudnn.benchmark = True
    n = (int)(((opt.scanner_no)*(opt.scanner_no - 1))/2)
    SSIM_sum = [0 for i in range(0, n)]
    MAE_sum = [0 for i in range(0, n)]
    image_count = 0
    
    for idx, (images, index) in enumerate(train_loader):

        image_count += len(index)
        image_no = images[0].shape[0]
        data_time.update_other_metrics(time.time() - end)
        
        if torch.cuda.is_available():
            for i in range(0, len(images)):
                images[i] = images[i].cuda(non_blocking=True)
        
        # compute loss
        embeddings, reconstructed_images = model(images, tarining_step_number) # tensor(btach-size * aug, features_dim)
        reconstructed_images = [torch.unsqueeze(reconstructed_image, dim = 1) for reconstructed_image in reconstructed_images]
        loss, loss1, loss2 = criterion(images, embeddings, reconstructed_images)

        # update metric
        train_loss.update_other_metrics(loss.item(), image_no)
        train_loss1.update_other_metrics(loss1.item(), image_no)
        train_loss2.update_other_metrics(loss2.item(), image_no)

        # optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update_other_metrics(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=train_loss))
            sys.stdout.flush()
    
    return train_loss.avg, train_loss1.avg, train_loss2.avg, SSIM_sum, MAE_sum


def set_loader_MISPEL(opt, train_adr_data, mask):

    # Loading the dataset.  
    dataset = HarmonizationDataSet(mask, train_adr_data, opt.scanner_no)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle = False,
        num_workers=opt.num_workers, pin_memory=True, sampler = None)
    return loader 
    

def set_loader(opt, train_adr_data, validation_adr_data, mask):

    # Loading the dataset.  
    train_dataset = HarmonizationDataSet(mask, train_adr_data, opt.scanner_no)
    validation_dataset = HarmonizationDataSet(mask, validation_adr_data, opt.scanner_no)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle = True,
        num_workers=opt.num_workers, pin_memory=True, sampler = None)
    
    valid_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=opt.batch_size, shuffle = True,
        num_workers=opt.num_workers, pin_memory=True, sampler = None)
    
    return train_loader, valid_loader


def read_mask(opt):   
    mask = nib.load(opt.mask_adr).get_fdata()
    return mask


def training(opt, optimizer, train_loader, valid_loader, model, criterion, logger, 
             tarining_step_number, mask, train_data, validation_data):

    # set parameters for the appropriate step
    if tarining_step_number == 1:
        epochs = opt.epochs_step1
        save_folder = opt.save_folder1
        save_freq = opt.save_freq_step1
        excel_filename_train = opt.output_excel_name_step1_train
        excel_filename_validation = opt.output_excel_name_step1_validation
        step = "Step1"
        data_frequency = opt.data_frequency_step1
    elif tarining_step_number == 2:
        epochs = opt.epochs_step2
        save_folder = opt.save_folder2
        save_freq = opt.save_freq_step2
        excel_filename_train = opt.output_excel_name_step2_train
        excel_filename_validation = opt.output_excel_name_step2_validation
        step = "Step2"
        data_frequency = opt.data_frequency_step2
        
    else:
        print("Invalid step number!")
    
    # train the network
    dataset_counter = 0
    for epoch in range(1, epochs + 1):
        set_seeds_torch(epoch)
        if ((epoch -1)% data_frequency == 0):
            if dataset_counter >  opt.Max_index_of_train_datasets:
                dataset_counter = 0
            substring = train_loader.dataset.adrs[0][0].split("/")[-2]
            train_loader.dataset.adrs = change_adrs(train_loader.dataset.adrs, substring , "train_nifti" + str(dataset_counter))
        
        ############ Train #############
        flag = False
        loss, loss1, loss2, SSIM, MAE = train(train_loader, model, criterion, optimizer, epoch, opt,
                                   tarining_step_number, mask, train_data, flag)
        print("Train loss: " + str(loss))

        # tensorboard logger
        logger.log_value(step + '_Train_loss', loss, epoch)
        logger.log_value(step + '_Train_loss1', loss1, epoch)
        logger.log_value(step + '_Train_loss2', loss2, epoch)
        logger.log_value(step + '_Train_learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # write to excel file
        write_into_excel_file(excel_filename_train, [epoch, loss, loss1, loss2])

        # save model
        if epoch %save_freq == 0:
                save_file = os.path.join(save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
                save_model(model, optimizer, opt, epoch, save_file)
                            
        ############ Validation  #############
        flag = False
        substring = valid_loader.dataset.adrs[0][0].split("/")[-2]      
        valid_loss, valid_loss1, valid_loss2, SSIM, MAE = validation(valid_loader, model, criterion, 
                                                          optimizer, epoch, opt, tarining_step_number, mask,
                                                           validation_data, flag)
        logger.log_value(step + '_Validation_loss', valid_loss, epoch)
        logger.log_value(step + '_Validation_loss1', valid_loss1, epoch)
        logger.log_value(step + '_Validation_loss2', valid_loss2, epoch)
        logger.log_value(step + '_Validation_learning_rate', optimizer.param_groups[0]['lr'], epoch) 
        write_into_excel_file(excel_filename_validation, [epoch, valid_loss, valid_loss1, valid_loss2])
     

def two_step_training(opt, train_loader, valid_loader, model, criterion1, criterion2, logger, 
                      mask, train_data, validation_data):
    
    print("###################################################################################")
    print("################################ Step1 ############################################")
    # training: step1
    time1000 = time.time()
    training_step = 1
    optimizer_step1 = set_optimizer(opt, model, training_step)
    training(opt, optimizer_step1, train_loader, valid_loader, model, criterion1,
              logger, training_step, mask, train_data, validation_data)
    time2000 = time.time()
       
    print("###################################################################################")
    print("################################ Step2 ############################################")
    # training: step2
    training_step = 2

    ''' freezing the encoder '''  
    for param in model.encoders.parameters():
        param.requires_grad = False
    optimizer_step2 = set_optimizer(opt, model, training_step)
    
    training(opt, optimizer_step2, train_loader, valid_loader, model, criterion2,
              logger, training_step, mask, train_data, validation_data)
    time3000 = time.time()
    print("step 1-start:" + str(time.asctime( time.localtime( time1000))))
    print("step 1-end:" + str(time.asctime( time.localtime( time2000))))
    print("step 2-end:" + str(time.asctime( time.localtime( time3000))))
                    

def Harmonization_method_main(gpu_number):
       
    opt = parse_option(gpu_number)

    # Read mask
    mask = read_mask(opt)

    train_data = pd.read_excel("./Dataset/Data_For_Loader_trainvalidation/saved_data/training_images_one_epoch_nifti_0.xlsx")
    validation_data = pd.read_excel("./Dataset/Data_For_Loader_trainvalidation/saved_data/validation_images_one_epoch_nifti_0.xlsx")

       
    # build data loader
    train_loader, validation_loader = set_loader(opt, train_data, validation_data, mask)
    set_seeds_torch(1024)
    
    # build model and criterion
    model = set_model(opt)
    
    # losses
    criterion1, criterion2 = set_losses(opt)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    
    # training 
    two_step_training(opt, train_loader, validation_loader, model, criterion1,
                       criterion2, logger, mask, train_data, validation_data)

