'''
Created on Jul 21, 2023

@author: MAE82
'''
import os
import pandas as pd



def validation_preparing_files_main(CV_no):
    
    dir= "./Dataset"
    excel_adr = "validation_images_one_epoch_nifti_" + CV_no + ".xlsx"
    adr_excel = os.path.join(dir, excel_adr)
    df = pd.read_excel(adr_excel)
    
    df.sort_values(by=['adr', 'slice'], inplace = True)
    df.reset_index(drop= True, inplace = True)
    adrs = df["adr"].tolist()
    adrs = [adr.split("/")[-1] for adr in adrs]
    df["image_name"] = adrs
    df.to_excel(os.path.join(dir, excel_adr.replace(".xlsx", "_sorted.xlsx")))
        
    all_names = df["adr"].tolist()
    names = []
    for name in all_names:
        val = name.split("/")[-2]
        if val in names:
            pass
        else:
            names.append(val)
            
    df = pd.read_excel(os.path.join(dir, "one_selected_scanner4_data_validation.xlsx"))        
    df["sorted_names"] = names
    df.sort_values(by=["sorted_names"], inplace = True)
    df.drop(columns = ["sorted_names"], inplace = True)
    df.to_excel(os.path.join(dir, "one_selected_scanner4_data_validation.xlsx"))

