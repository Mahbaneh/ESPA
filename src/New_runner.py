'''
Created on Sep 30, 2023

@author: MAE82
'''

from New_main_supcon import main, main_with_sliceIndex
import time
import os


# Setting GPUs
gpu_number = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number


Start = time.time()
main(gpu_number)
#main_with_sliceIndex(gpu_number)
print("finished the run!")
end = time.time()
print("Startt:" + str(time.asctime(time.localtime(Start))))
print("End:" + str(time.asctime( time.localtime(end))))

'''
scanners are: ["ge", "philips", "prisma", "trio"]
python3.8 New_runner.py 
'''
