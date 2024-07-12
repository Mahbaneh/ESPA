
# ESPA: An Unsupervised Harmonization Framework via Enhanced Structure Preserving Augmentation
# Reference to paper: 
Method | Citation | Links 
--- | --- | --- 
ESPA | Eshaghzadeh Torbati M., Minhas D. S., Tafti A. P., DeCarli C. S., Tudorascu D. L., and Hwang S. J., 2024, October. ESPA: An Unsupervised Harmonization Framework via Enhanced Structure Preserving Augmentation. In International Conference on Medical Image Computing and Computer-Assisted Intervention. | [Paper](TBD) [Poster](TBD)
# Table of content:
[Software requirements](#Software-requirements)\
[ESPA harmonization](#ESPA-Harmonization)\
[Structure of input data for ESPA](#Structure-of-input-data-for-ESPA)\
[Image preprocessing](#Image-Preprocessing)\
[Running](#Running)

# Software requirements:
Python and Pytorch. 

# ESPA harmonization: 
ESPA, crafted as an unsupervised task-agnostic image-harmonization framework, adapts images to a scanner-middle-ground domain. In achieving this adaptation, we employed our harmonization technique, [MISPEL](https://github.com/Mahbaneh/MISPEL/tree/main), albeit with a significant modification. Instead of relying on matched data, we introduce a novel approach wherein we simultaneously generate and utilize simulated matched images during the training of MISPEL. This approach equips MISPEL with simulated data of substantial size, offering a solution to the model robustness challenge in harmonization, particularly concerning supervised harmonization methods. 
The simulated matched images are generated using our novel structure-preserving augmentation methods: (1) tissue-type contrast augmentation, and (2) GAN-based residual augmentation. Initially, each augmentation method is individually configured to adapt images of a _source scanner_ to those of the _target scanners_. We refer to the data targeted for harmonization
as _multi-scanner_ data. This data contains images of _M_ scanners. We consider another set of data with images of one arbitrary scanner and refer to it as _source_
data. We refer to scanners of the source and multi-scanner data as the _source_ scanner and _target_ scanners, respectively.

![This is an image](https://github.com/Mahbaneh/ESPA/blob/main/ESPA_Framework.png)
# Tissue type contrast augmentation: 
![This is an image](https://github.com/Mahbaneh/ESPA/blob/main/ESPA_TC1.png)
![This is an image](https://github.com/Mahbaneh/ESPA/blob/main/ESPA_TC2.png)
# GAN-based residual augmentation: 
![This is an image](https://github.com/Mahbaneh/ESPA/blob/main/ResidualGAN.png)

# Structure of input data for ESPA:
TBD.

# Image preprocessing:
For the preprocessing step please read the 'preprocessing' paragraph in section 3 of our paper. The steps are: (1) Registration to a template, (2) N4 bias correction, (3) Skull stripping, and (4) Image scaling.
For the first three steps, we used the instruction prepared in the [RAVEL repositoty](https://github.com/Jfortin1/RAVEL). 

# Running:
# ESPA_TC: ESPA trained using tissue-type contrast augmentation
For training  
```
python3 main.py --data_dir "Data" --mask_adr 'Data/JHU_MNI_SS_T1_Brain_Mask.nii' \
--output_dir 'Data/Output' --downsample False --normalizing False \
--upsampling False  --Swap_axis True \
--latent_dim 6 --batch_size 4 --learning_rate 0.0001 \
--T1 100 --T2 100  --scanner_names "ge,philips,trio,prisma"\
--lambda1 0.3 --lambda2 1.0 --lambda3 1.0 --lambda4 4.0
```
For training ESPA using tissue-type contrast augmentation.
```
python3 main.py --data_dir "Data" --mask_adr 'Data/JHU_MNI_SS_T1_Brain_Mask.nii' \
--output_dir 'Data/Output' --downsample False --normalizing False \
--upsampling False  --Swap_axis True \
--latent_dim 6 --batch_size 4 --learning_rate 0.0001 \
--T1 100 --T2 100  --scanner_names "ge,philips,trio,prisma"\
--lambda1 0.3 --lambda2 1.0 --lambda3 1.0 --lambda4 4.0
```
# ESPA_TC: ESPA trained using GAN-based residual augmentation
```
python3 main.py --data_dir "Data" --mask_adr 'Data/JHU_MNI_SS_T1_Brain_Mask.nii' \
--output_dir 'Data/Output' --downsample False --normalizing False \
--upsampling False  --Swap_axis True \
--latent_dim 6 --batch_size 4 --learning_rate 0.0001 \
--T1 100 --T2 100  --scanner_names "ge,philips,trio,prisma"\
--lambda1 0.3 --lambda2 1.0 --lambda3 1.0 --lambda4 4.0
```
For training ESPA using GAN-based residual augmentation.
```
python3 main.py --data_dir "Data" --mask_adr 'Data/JHU_MNI_SS_T1_Brain_Mask.nii' \
--output_dir 'Data/Output' --downsample False --normalizing False \
--upsampling False  --Swap_axis True \
--latent_dim 6 --batch_size 4 --learning_rate 0.0001 \
--T1 100 --T2 100  --scanner_names "ge,philips,trio,prisma"\
--lambda1 0.3 --lambda2 1.0 --lambda3 1.0 --lambda4 4.0
```
