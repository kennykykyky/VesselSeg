import glob
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pdb
import numpy as np
import nibabel as nib

def rename_COWdata(datapath, label = False):
    
    # rename the data inside datapath to '<name>_xxx_0000.nii.gz', where xxx is the id number for each case and it starts from 001 to the total number of cases
    # e.g. rename_nnUnetdata('./data/CAS2020_training', name='TOF')
    
    paths_items = glob.glob(os.path.join(datapath, '*.nii.gz'))
    paths_items.sort()

    for i, path_item in enumerate(paths_items):
        name_item = os.path.basename(path_item)
        if not label:
            name_item_new = f'{i+1:03d}.nii.gz'
        os.rename(path_item, os.path.join(os.path.dirname(path_item), name_item_new))
        print(f'{name_item} -> {name_item_new}')
        
datapath = '/home/kaiyu/project/VesselSeg/data/TopCoW_detection/image'
# rename_COWdata(datapath, label = False)

def rename_COWlabel(datapath):

    paths_items = glob.glob(os.path.join(datapath, '*.png'))
    paths_items.sort()

    for i, path_item in enumerate(paths_items):
        name_item = os.path.basename(path_item)
        name_item_new = f'{i+1:03d}.png'
        os.rename(path_item, os.path.join(os.path.dirname(path_item), name_item_new))
        print(f'{name_item} -> {name_item_new}')

bbox_path  = '/home/kaiyu/project/VesselSeg/data/TopCoW_detection/bbox'
rename_COWlabel(bbox_path)

pdb.set_trace()
