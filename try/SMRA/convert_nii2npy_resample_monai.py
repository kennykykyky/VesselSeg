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
    meta_file = pd.read_csv('./data/CAS2023_training/meta.csv')
    subdirs = ['imagesTr', 'imagesTs', 'labelsTr', 'labelsTs']

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    datapath = '/home/kaiyu/project/VesselSeg/data/SMRA_MONAI_nii_resample/MICCAI_CAS_2023_extra.json'
    with open(os.path.join('/home/kaiyu/project/nnUNet_dataset/nnUNet_results/Dataset305_SMRAResample/nnUNetTrainer__nnUNetPlans__3d_fullres', 'plans.json'), 'r') as f:
        plans = json.load(f)

    # Define target spacing
    target_spacing = plans['configurations']['3d_fullres']['spacing']
    # move the first element to the last
    target_spacing = np.roll(target_spacing, 2)

    # Create a dictionary dataset
    # data_dicts = [
    #     {"image": image_file, "label": label_file, "old_spacing": old_spacing}
    #     for image_file, label_file, old_spacing in zip(image_files, label_files, old_spacings)
    # ]

    # Create transformations
    transforms = Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=False),
        Spacingd(keys=["image", "label"], pixdim=target_spacing, mode=("bilinear", "nearest"), align_corners=True),
    ])

    # transforms = [LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=False)]

    img_save = ['imagesTr', 'imagesTr', 'imagesTs']
    label_save = ['labelsTr', 'labelsTr', 'labelsTs']
    for i, task in enumerate(['training', 'validation', 'test']):

        data_dicts = load_decathlon_datalist(datapath, True, task)

        # Create dataset and data loader
        dataset = Dataset(data=data_dicts, transform=transforms)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        # Process each file
        for data in loader:
            # Get image data from the batch
            image_data = data["image"].squeeze()
            label_data = data["label"].squeeze()

            # Define numpy save path
            base_name = os.path.basename(data["image_meta_dict"]["filename_or_obj"][0])[0:-7]

            # read the original nii file and print the shape
            # img = sitk.ReadImage(data["image_meta_dict"]["filename_or_obj"][0])
            # print(img.GetSize())

            img_save_path = os.path.join(os.path.join(output_folder, img_save[i]), f"{base_name}.npy")
            label_save_path = os.path.join(os.path.join(output_folder, label_save[i]), f"{base_name}.npy")

            # Save as numpy array
            np.save(img_save_path, image_data)
            np.save(label_save_path, label_data)

# Use the function
convert_nifti_to_numpy('/home/kaiyu/project/VesselSeg/data/SMRA_MONAI_nii_resample', '/home/kaiyu/project/VesselSeg/data/SMRA_MONAI_npy_resample')
# convert_nifti_to_numpy('/home/kaiyu/project/VesselSeg/data/CAS2023_test', '/home/kaiyu/project/VesselSeg/data/SMRA_MONAI_test_npy')
