import glob
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pdb
import numpy as np
import nibabel as nib
import SimpleITK as sitk


def rename_nnUnetdata(datapath, name='TOF', label = False):
    
    # rename the data inside datapath to '<name>_xxx_0000.nii.gz', where xxx is the id number for each case and it starts from 001 to the total number of cases
    # e.g. rename_nnUnetdata('./data/CAS2020_training', name='TOF')
    
    paths_items = glob.glob(os.path.join(datapath, '*.nii.gz'))
    paths_items.sort()
    for i, path_item in enumerate(paths_items):
        name_item = os.path.basename(path_item)
        if not label:
            name_item_new = f'{name}_{i+1:03d}_0000.nii.gz'
        else:
            name_item_new = f'{name}_{i+1:03d}.nii.gz'
        os.rename(path_item, os.path.join(os.path.dirname(path_item), name_item_new))
        print(f'{name_item} -> {name_item_new}')
        
rename_nnUnetdata('/home/kaiyu/project/nnUNet_dataset/nnUNet_raw/Dataset305_SMRAResample/labelsTr', name='TOF')
datapath = '/home/kaiyu/project/nnUNet_dataset/nnUNet_raw/Dataset305_SMRAResample/labelsTr'
rename_nnUnetdata(datapath, name='TOF', label = True)

pdb.set_trace()



path_dataset='./data/CAS2023_training/mask'

paths_items_all = glob.glob(os.path.join(path_dataset, '*.nii.gz'))
paths_items_avail = []

meta_file = pd.read_csv('./data/CAS2023_training/meta.csv')

for path_item in paths_items_all[:]:

    print(path_item)
    
    name_item = os.path.basename(path_item)
    id_item = int(name_item[:-7])
    
    # pos_s = meta_file.loc[meta_file['index'] == id_item, 'pos_s'].values[0]
    # pos_s = pos_s.strip('[]')
    # bbox_list = pos_s.split(',')
    # pos_s = np.array(bbox_list, dtype=int)

    spacing = meta_file.loc[meta_file['index'] == id_item, 'spacing'].values[0]
    spacing = spacing.strip('()')
    spacing_list = spacing.split(',')
    spacing = np.array(spacing_list, dtype=float)

    path_img = os.path.join(path_dataset, f'{name_item}')
    path_seg = os.path.join(path_dataset, f'{name_item}')

    # img = nib.load(path_img)
    # seg = nib.load(path_seg)

    # img = sitk.ReadImage(path_img)
    seg = sitk.ReadImage(path_seg)
    # img.SetSpacing(spacing)
    seg.SetSpacing(spacing)
    path_img_new = path_img.replace('training/data', 'training/data_new')
    path_seg_new = path_seg.replace('training/mask',  'training/mask_new')
    # sitk.WriteImage(img, path_img_new)
    sitk.WriteImage(seg, path_seg_new)

    # times minus 1 for the first two components of spacing
    # spacing[:2] = spacing[:2] * -1

    # save the spacing info to the header of nifti file
    # img.affine[:3, :3] = np.diag(spacing)
    # seg.affine[:3, :3] = np.diag(spacing)

    # path_img_new = path_img.replace('training/data', 'training/data_new')
    # path_seg_new = path_seg.replace('training/mask', 'training/mask_new')

    # # save the nifti file
    # nib.save(img, path_img_new)
    # nib.save(seg, path_seg_new)




# End of plotting SMRA-MICCAI data
    
pdb.set_trace()
    