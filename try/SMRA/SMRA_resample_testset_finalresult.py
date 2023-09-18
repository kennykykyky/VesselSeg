import glob
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pdb
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from monai.transforms import Spacingd, ToNumpyD, LoadImaged, Compose
from monai.data import Dataset, DataLoader, load_decathlon_datalist

path_dataset='/home/kaiyu/project/VesselSeg/save/test_03_14_resample_test_2023-08-21_09-03-52'
datapath = '/home/kaiyu/project/VesselSeg/save/test_03_14_resample_test_2023-08-21_09-03-52/test.json'
output_folder = '/home/kaiyu/project/VesselSeg/save/test_03_14_resample_test_2023-08-21_09-03-52/resample'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

paths_items_all = glob.glob(os.path.join(path_dataset, '*.nii.gz'))
paths_items_avail = []

meta_file = pd.read_csv('/home/kaiyu/project/VesselSeg/data/CAS2023_training/meta_with_nf.csv')
# meta_file = pd.read_csv('./data/CAS2023_test/meta.csv')

for path_item in paths_items_all[:]:

    print(path_item)
    
    name_item = os.path.basename(path_item)
    id_item = int(name_item[:-7])

    spacing = meta_file.loc[meta_file['index'] == id_item, 'spacing'].values[0]
    spacing = spacing.strip('()')
    spacing_list = spacing.split(',')
    spacing = np.array(spacing_list, dtype=float)

    # target_spacing = np.roll(spacing, 2)

    # Create transformations
    transforms = Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=False, allow_missing_keys=True),
        Spacingd(keys=["image", "label"], pixdim=spacing, mode=("bilinear", "nearest"), align_corners=True, allow_missing_keys=True),
    ])

    data_dicts = load_decathlon_datalist(datapath, True, 'test')

    # Create dataset and data loader
    dataset = Dataset(data=data_dicts, transform=transforms)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    pdb.set_trace()

    # Process each file
    for data in loader:
        # Get image data from the batch
        image_data = data["image"].squeeze()

        # Define numpy save path
        base_name = os.path.basename(data["image_meta_dict"]["filename_or_obj"][0])[0:-7]

        # read the original nii file and print the shape
        # img = sitk.ReadImage(data["image_meta_dict"]["filename_or_obj"][0])
        # print(img.GetSize())

        img_save_path = os.path.join(output_folder, f"{base_name}.nii.gz")

        # save the image_data as nii file, which the spacing should set to (1, 1, 1)
        img = sitk.GetImageFromArray(image_data)
        img.SetSpacing((1, 1, 1))
        sitk.WriteImage(img, img_save_path)

pdb.set_trace()
    