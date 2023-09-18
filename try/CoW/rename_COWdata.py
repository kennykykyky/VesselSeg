import glob
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pdb
import numpy as np
import nibabel as nib

def rename_COWdata(datapath, name='TOF', label = False):
    
    # rename the data inside datapath to '<name>_xxx_0000.nii.gz', where xxx is the id number for each case and it starts from 001 to the total number of cases
    # e.g. rename_nnUnetdata('./data/CAS2020_training', name='TOF')
    
    paths_items = glob.glob(os.path.join(datapath, '*.nii.gz'))
    paths_items.sort()

    # detect whether the paths_items already have the correct name, if so, calculate the number of cases and then remove them from paths_items
    paths_items_new = []
    for path_item in paths_items:
        name_item = os.path.basename(path_item)
        if name_item.split('_')[0] == name:
            paths_items_new.append(path_item)
    start_num = len(paths_items_new)
    paths_items = list(set(paths_items) - set(paths_items_new))
    paths_items.sort()

    for i, path_item in enumerate(paths_items):
        name_item = os.path.basename(path_item)
        if not label:
            name_item_new = f'{name}_{i+1+start_num:03d}_0000.nii.gz'
        else:
            name_item_new = f'{name}_{i+1+start_num:03d}.nii.gz'
        os.rename(path_item, os.path.join(os.path.dirname(path_item), name_item_new))
        print(f'{name_item} -> {name_item_new}')
        
datapath = '/home/kaiyu/project/nnUNet_dataset/nnUNet_raw/Dataset304_COWMRABI/labelsTr'
rename_COWdata(datapath, name='COWMRROI', label = True)

pdb.set_trace()
