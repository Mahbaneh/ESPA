
# ESPA: An Unsupervised Harmonization Framework via Enhanced Structure Preserving Augmentation
# Reference to paper: 
Method | Citation | Links 
--- | --- | --- 
ESPA | TBD. | [Paper](TBD) [Poster](TBD) [Presentation](TBD)
# Table of content:
[Software requirements](#Software-requirements)\
[ESPA harmonization](#ESPA-Harmonization)\
[Structure of input data for ESPA](#Structure-of-input-data-for-ESPA)\
[Image preprocessing](#Image-Preprocessing)\
[Running](#Running)

# Software requirements:
Python and Pytorch. 

# ESPA harmonization: 
TBD.
![This is an image](https://github.com/Mahbaneh/ESPA/blob/main/ResidualGAN.png)

# Structure of input data for ESPA:
TBD.

# Image preprocessing:
TBD.

# Running:
TBD.
```
python3 main.py --data_dir "Data" --mask_adr 'Data/JHU_MNI_SS_T1_Brain_Mask.nii' \
--output_dir 'Data/Output' --downsample False --normalizing False \
--upsampling False  --Swap_axis True \
--latent_dim 6 --batch_size 4 --learning_rate 0.0001 \
--T1 100 --T2 100  --scanner_names "ge,philips,trio,prisma"\
--lambda1 0.3 --lambda2 1.0 --lambda3 1.0 --lambda4 4.0
```
