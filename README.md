
# ESPA: An Unsupervised Harmonization Framework via Enhanced Structure Preserving Augmentation
# Reference to paper: 
Method | Citation | Links 
--- | --- | --- 
ESPA | Eshaghzadeh Torbati M., Minhas D. S., Tafti A. P., DeCarli C. S., Tudorascu D. L., and Hwang S. J., 2024, October. ESPA: An Unsupervised Harmonization Framework via Enhanced Structure Preserving Augmentation. In International Conference on Medical Image Computing and Computer-Assisted Intervention. | [Paper](TBD) [Poster](TBD)
# Table of content:
[Software requirements](#Software-requirements)\
[ESPA framework](#ESPA-framework)\
[Structure of input data for ESPA](#Structure-of-input-data-for-ESPA)\
[Image preprocessing](#Image-Preprocessing)\
[Running](#Running)

# Software requirements:
Python and Pytorch. 

# ESPA framework: 
ESPA, crafted as an unsupervised task-agnostic image-harmonization framework, adapts images to a scanner-middle-ground domain. In achieving this adaptation, we employed our harmonization technique, [MISPEL](https://github.com/Mahbaneh/MISPEL/tree/main), albeit with a significant modification. Instead of relying on matched data, we introduce a novel approach wherein we simultaneously generate and utilize simulated matched images during the training of MISPEL. This approach equips MISPEL with simulated data of substantial size, offering a solution to the model robustness challenge in harmonization, particularly concerning supervised harmonization methods. 
The simulated matched images are generated using our novel structure-preserving augmentation methods: (1) tissue-type contrast augmentation, and (2) GAN-based residual augmentation. Initially, each augmentation method is individually configured to adapt images of a _source scanner_ to those of the _target scanners_. We refer to the data targeted for harmonization
as _multi-scanner_ data. This data contains images of _M_ scanners. We consider another set of data with images of one arbitrary scanner and refer to it as _source_
data. We refer to scanners of the source and multi-scanner data as the _source_ scanner and _target_ scanners, respectively.

![This is an image](https://github.com/Mahbaneh/ESPA/blob/main/ESPA_Framework.png)
# Tissue type contrast augmentation: 
Scanner effects can impact brain tissue contrast. To address this issue, we utilize a three-step augmentation approach aimed at adjusting tissue contrast from a source scanner to a target scanner while maintaining brain structure. This method builds upon previous work by [paper](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.708196/full) initially developed for brain segmentation purposes. It is important to note that this augmentation technique adapts images from the source scanner to a _single_ target scanner. Therefore, for each of the _M_ target scanners, we should configure _M_ distinct tissue-type contrast augmentation methods.
![This is an image](https://github.com/Mahbaneh/ESPA/blob/main/ESPA_TC1.png)
![This is an image](https://github.com/Mahbaneh/ESPA/blob/main/ESPA_TC2.png)
# GAN-based residual augmentation: 
Scanner effects can be more intricate than tissue-type modifications and can vary across brain regions. Thus, we develop a GAN-based augmentation method to generate and sample scanner effects as images (_residuals_) added to the original images. By limiting scanner effects as additive components to images, we consider brain structure during augmentation. For this purpose, we introduce Residual StarGAN, which performs image-to-image translation between all pairs of our scanner domains: source and target scanners.
![This is an image](https://github.com/Mahbaneh/ESPA/blob/main/ResidualGAN.png)

# Structure of input data for ESPA:
Matched data in unmatched format. Not necessarily should be matched. And OASIS ... 

# Image preprocessing:
For the preprocessing step please read the 'preprocessing' paragraph in section 3 of our paper. The steps are: (1) Registration to a template, (2) N4 bias correction, (3) Skull stripping, and (4) Image scaling.
For the first three steps, we used the instruction prepared in the [RAVEL repositoty](https://github.com/Jfortin1/RAVEL). 

# Running:
# ESPA_TC: ESPA trained using tissue-type contrast augmentation
Configuring tissue-type contrast augmentation:


For training ESPA using tissue-type contrast augmentation.

# ESPA_Res: ESPA trained using GAN-based residual augmentation
**Configuring GAN-based residual augmentation:**

There are three **input arguments** for input data: 1) external_scanner_image_adrs, 2) target_scanner_image_adrs, 2) CVfolds_adrs. 1) _external_scanner_image_adrs_ is the address for images of the _source_ scanner (_source_ data). 2) _target_scanner_image_adrs_ is the address to the directory of images for the source scanners (_multi-scanner_ data). 3) _CVfolds_adrs_ is the address to the list of cross-validated images, if we used cross-validation for the multi-scanner data. Running the following command, a new **output folder** is created as "save" containing the trained residual-StarGAN models. The trained generator is then used for generating augmented images. The augmented images are then used for training ESPA_Res.
```
python Residual_StarGAN.py --n_epochs 200 --CV_no 2 --batch_size 64 --lr_Gen 0.0002\
--lr_Dsc 0.0002 --nz 192 --target_scanner_image_adrs\
"./Dataset/ResGAN_Configuration/TargetScanners"\
--external_scanner_image_adrs "./Dataset/ResGAN_Configuration/ExternalScanner" --CVfolds_adrs\
 "./Dataset/ResGAN_Configuration/CV_Folds" --b1 0.5 --b2 0.999 --checkpoint_interval 20
```
**Training ESPA using GAN-based residual augmentation:**

There are three **input arguments** for input data: 1) data_dir, 2) train_excel_adr, and 3) aug_data_folder. 1) _data_dir_ is the directory for the data used in this step. 2) _train_excel_adr_ is the directory to the list of augmented images (the data generated in Configuring GAN-based residual augmentation). 3) _aug_data_folder_ is the directory for the augmented images. Running the following command, a new **output folder** is created as "save" containing the trained ESPA_Res model. This model is then used for harmonizing unseen images of target scanners directly. 
```
python Training_ESPA_Res.py --lambda1 1.0 --lambda2 200.0 --lambda3 1.0 --lambda4 1.0\
 --no_latent_embeddings 6 --epochs_step1 100 --epochs_step2 400 --learning_rate_step1 0.0001\
 --learning_rate_step2 0.00001 --num_workers 0 --data_frequency_step2 5 --data_frequency_step2 14\
 --save_freq_step1 50 --save_freq_step2 50  --c_dim 5 --res_blocks 9 --nz 192 --data_dir\
 "./Dataset/ESPA_Res_Training/ErternalScannerTrainingMISPEL" --train_excel_adr\
 "./Dataset/ESPA_Res_Training/ErternalScannerTrainingMISPEL/OASIS_train.xlsx"\
 --aug_data_folder "Augmented_train_data"
```
