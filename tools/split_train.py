# Imports
import random
import os
import shutil
random.seed(1000)
root = "/home/helen/Documents/datasets/frankascan-recenter"
split = 'train'
data_root = os.path.join(root,split)
files = os.listdir(data_root)
random.shuffle(files)

split = 0.2
num_val = int(split*len(files))

for validation_file in files[0:num_val]:
    original = os.path.join(data_root,validation_file)
    target = os.path.join(root,'val',validation_file)
    shutil.move(original,target)
