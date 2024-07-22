'''
Created on Jul 5, 2022

@author: MAE82
'''
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os


def main_step2(dir, dir_out):
    

    train_adr = os.path.join(dir, "training_step2/train_epoches.xlsx") 
    validation_adr = os.path.join(dir, "training_step2/validation_epoches.xlsx")
    

    ######################### train #########################
    data_train = pd.read_excel(train_adr)
    data_train["Loss2"] = data_train["Loss2"] 
    
    plt.figure()    
    ax = sns.lineplot(data = data_train, y = "Loss", x = "Epochs", color = 'b')
    ax = sns.lineplot(data = data_train, y = "Loss1", x = "Epochs", color = 'g' )
    ax = sns.lineplot(data = data_train, y = "Loss2", x = "Epochs", color = 'r')

   
    ax.set_title('Training: L1(recon)= 0.3, L2(coup)= 1.0')
    ax.set_ylabel("Avg-loss")
    plt.legend(labels=["Step1_loss","Step2_recon_loss", "Step1_harmonization_loss"])
    plt.savefig(os.path.join(dir_out, "step2_Fig1.png"))
    
    ######################### validation #########################
    data_validation = pd.read_excel(validation_adr)
    data_validation["Loss2"] = data_validation["Loss2"]

    plt.figure()
    ax = sns.lineplot(data = data_validation, y = "Loss", x = "Epochs", color = 'b')
    ax = sns.lineplot(data = data_validation, y = "Loss1", x = "Epochs", color ='g')
    ax = sns.lineplot(data = data_validation, y = "Loss2", x = "Epochs", color ='r')
    
    ax.set_title('Validation: L1(recon)= 0.3, L2(coup)= 1.0')
    ax.set_ylabel("Avg-loss")
    plt.legend(labels=["Step1_loss","Step1_recon_loss", "Step1_harmonization_loss"])
    plt.savefig(os.path.join(dir_out, "step2_Fig2.png"))
    
    ######################### train and validation #########################
    data_train = pd.read_excel(train_adr)
    data_validation = pd.read_excel(validation_adr)
    
    plt.figure()
    ax = sns.lineplot(data = data_train, y = "Loss", x = "Epochs", color = 'b')
    ax = sns.lineplot(data = data_validation, y = "Loss", x = "Epochs", color = 'b', linestyle="dashed")
    
    
    ax.set_title('Training_validation: L1(recon)= 0.3, L2(coup)= 1.0')
    ax.set_ylabel("Avg-loss")
    plt.legend(labels=["Train: Step1_loss","Validation: Step1_loss"])
    plt.savefig(os.path.join(dir_out, "step2_Fig3.png"))



def main_step1(dir, dir_out):
    

    train_adr = os.path.join(dir, "training_step1/train_epoches.xlsx") 
    validation_adr = os.path.join(dir, "training_step1/validation_epoches.xlsx") 
    

    ######################### train #########################
    data_train = pd.read_excel(train_adr)
    data_train["Loss2"] = data_train["Loss2"] 
    
    plt.figure()   
    ax = sns.lineplot(data = data_train, y = "Loss", x = "Epochs", color = 'b')
    ax = sns.lineplot(data = data_train, y = "Loss1", x = "Epochs", color = 'g' )
    ax = sns.lineplot(data = data_train, y = "Loss2", x = "Epochs", color = 'r')

   
    ax.set_title('Training: L1(recon)= 0.3, L2(coup)= 1.0')
    ax.set_ylabel("Avg-loss")
    plt.legend(labels=["Step1_loss","Step1_recon_loss", "Step1_coupling_loss * 100"])
    plt.savefig(os.path.join(dir_out, "step1_Fig1.png"))

    
    ######################### validation #########################
    data_validation = pd.read_excel(validation_adr)
    data_validation["Loss2"] = data_validation["Loss2"]

    plt.figure()  
    ax = sns.lineplot(data = data_validation, y = "Loss", x = "Epochs", color = 'b')
    ax = sns.lineplot(data = data_validation, y = "Loss1", x = "Epochs", color ='g')
    ax = sns.lineplot(data = data_validation, y = "Loss2", x = "Epochs", color ='r')
    
    ax.set_title('Validation: L1(recon)= 0.3, L2(coup)= 1.0')
    ax.set_ylabel("Avg-loss")
    plt.legend(labels=["Step1_loss","Step1_recon_loss", "Step1_coupling_loss"])
    plt.savefig(os.path.join(dir_out, "step1_Fig2.png"))
    
    
    ######################### train and validation #########################
    data_train = pd.read_excel(train_adr)
    data_validation = pd.read_excel(validation_adr)
    
    plt.figure()  
    ax = sns.lineplot(data = data_train, y = "Loss", x = "Epochs", color = 'b')
    ax = sns.lineplot(data = data_validation, y = "Loss", x = "Epochs", color = 'b', linestyle="dashed")
    
    
    ax.set_title('Training_validation: L1(recon)= 0.3, L2(coup)= 1.0')
    ax.set_ylabel("Avg-loss")
    plt.legend(labels=["Train: Step1_loss","Validation: Step1_loss"])
    plt.savefig(os.path.join(dir_out, "step1_Fig3.png"))
    

def main_plotting_losses(save_number, model_name, folder_name):

    print("start plotting losses!")
    dir = save_number + "/SupCon/cifar10_models/" + model_name
    dir_out = save_number + "/Results/" + folder_name + "/" 
    main_step1(dir, dir_out)
    main_step2(dir, dir_out)
    print("Finished plotting losses!")


