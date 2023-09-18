import glob
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pdb
import numpy as np
import nibabel as nib
import SimpleITK as sitk

path_dataset='./data/CAS2023_test/imagesTs'

paths_items_all = glob.glob(os.path.join(path_dataset, '*.nii.gz'))
paths_items_avail = []

meta_file = pd.read_csv('./data/CAS2023_test/meta.csv')

for path_item in paths_items_all[:]:

    print(path_item)
    
    name_item = os.path.basename(path_item)
    id_item = int(name_item[:-7])

    spacing = meta_file.loc[meta_file['index'] == id_item, 'spacing'].values[0]
    spacing = spacing.strip('()')
    spacing_list = spacing.split(',')
    spacing = np.array(spacing_list, dtype=float)

    path_img = os.path.join(path_dataset, f'{name_item}')

    img = sitk.ReadImage(path_img)
    img.SetSpacing(spacing)
    path_img_new = path_img.replace('test/imagesTs', 'test/imagesTs_resample')
    sitk.WriteImage(img, path_img_new)


pdb.set_trace()
    