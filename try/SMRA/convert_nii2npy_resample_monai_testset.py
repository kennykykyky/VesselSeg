import os
import numpy as np
import json
from monai.transforms import Spacingd, ToNumpyD, LoadImaged, Compose
from monai.data import Dataset, DataLoader, load_decathlon_datalist
import SimpleITK as sitk
import pdb
import pandas as pd

def convert_nifti_to_numpy(input_folder, output_folder):
    # Define subdirectories
    # subdirs = ['imagesTs']
    meta_file = pd.read_csv('./data/CAS2023_test/meta.csv')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    datapath = '/home/kaiyu/project/VesselSeg/data/CAS2023_test/MICCAI_CAS_2023_test.json'
    with open(os.path.join('/home/kaiyu/project/nnUNet_dataset/nnUNet_results/Dataset305_SMRAResample/nnUNetTrainer__nnUNetPlans__3d_fullres', 'plans.json'), 'r') as f:
        plans = json.load(f)

    # Define target spacing
    target_spacing = plans['configurations']['3d_fullres']['spacing']
    # move the first element to the last
    target_spacing = np.roll(target_spacing, 2)

    # Create transformations
    transforms = Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=False, allow_missing_keys=True),
        Spacingd(keys=["image", "label"], pixdim=target_spacing, mode=("bilinear", "nearest"), align_corners=True, allow_missing_keys=True),
    ])

    # transforms = [LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=False)]

    img_save = ['imagesTs']
    for i, task in enumerate(['test']):

        data_dicts = load_decathlon_datalist(datapath, True, task)

        # Create dataset and data loader
        dataset = Dataset(data=data_dicts, transform=transforms)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        # Process each file
        for data in loader:
            # Get image data from the batch
            image_data = data["image"].squeeze()

            # Define numpy save path
            base_name = os.path.basename(data["image_meta_dict"]["filename_or_obj"][0])[0:-7]

            # read the original nii file and print the shape
            # img = sitk.ReadImage(data["image_meta_dict"]["filename_or_obj"][0])
            # print(img.GetSize())

            img_save_path = os.path.join(os.path.join(output_folder, img_save[i]), f"{base_name}.npy")

            # Save as numpy array
            np.save(img_save_path, image_data)

# Use the function
convert_nifti_to_numpy('/home/kaiyu/project/VesselSeg/data/CAS2023_test', '/home/kaiyu/project/VesselSeg/data/SMRA_MONAI_test_npy_resample')
