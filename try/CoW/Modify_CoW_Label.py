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
    for i, path_item in enumerate(paths_items):
        name_item = os.path.basename(path_item)
        if name not in name_item:
            if not label:
                name_item_new = f'{name}_{i+1:03d}_0000.nii.gz'
            else:
                name_item_new = f'{name}_{i+1:03d}.nii.gz'
            os.rename(path_item, os.path.join(os.path.dirname(path_item), name_item_new))
            print(f'{name_item} -> {name_item_new}')
        
def rename_nnUnet2Monai_data(datapath):
    
    paths_items = glob.glob(os.path.join(datapath, '*.nii.gz'))
    paths_items.sort()
    for i, path_item in enumerate(paths_items):
        itemid = os.path.basename(path_item).split('_')[1]
        name_item_new = f'{itemid}.nii.gz'
        os.rename(path_item, os.path.join(os.path.dirname(path_item), name_item_new))
        print(f'{itemid} -> {name_item_new}')

# datapath = '/home/kaiyu/project/VesselSeg/data/CoW_Semantic_ROI/imagesTr'
# labelpath = '/home/kaiyu/project/VesselSeg/data/CoW_Semantic_ROI/labelsTr'
# rename_nnUnet2Monai_data(datapath)
# rename_nnUnet2Monai_data(labelpath)
# pdb.set_trace()

# datapath = '/home/kaiyu/project/nnUNet_dataset/nnUNet_raw/Dataset302_COWMRA/labelsTr'
# rename_COWdata(datapath, name='COWMRROI', label = True)
# pdb.set_trace()

label_file = '/home/kaiyu/project/nnUNet_dataset/nnUNet_raw/Dataset302_COWMRA/labelsTr'
label_files = glob.glob(os.path.join(label_file, '*.nii.gz'))

for path_item in label_files:
    seg = nib.load(path_item)
    seg_data = seg.get_fdata()
    print(seg_data.max())
    old_data = seg_data.copy()
    
    print(path_item.split('\\')[-1])
    # my_list = list(range(13)) + [15]
    # print(list(set(my_list) - set(np.unique(seg_data))))
    
    # replace the pixels with value 15 to 13
    seg_data[seg_data == 15] = 13
    print(np.unique(seg_data))
    
    # overwrite the original segmentation mask in nii format, the data should be integer
    seg_new = nib.Nifti1Image(seg_data.astype(np.uint8), seg.affine, seg.header)
    
    # save the nii file
    nib.save(seg_new, path_item)
    
    # test = nib.load(path_item.replace('001', '001_test'))
    # print(np.unique(test.get_fdata()))
    # pdb.set_trace()


pdb.set_trace()
