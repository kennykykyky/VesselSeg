import os
import numpy as np
import nibabel as nib

def convert_nifti_to_numpy(input_folder, output_folder):
    # Define subdirectories
    # subdirs = ['imagesTs']
    subdirs = ['imagesTr', 'imagesTs', 'labelsTr', 'labelsTs']

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over each subdirectory
    for subdir in subdirs:
        input_subdir = os.path.join(input_folder, subdir)
        output_subdir = os.path.join(output_folder, subdir)

        # Make sure the output subdirectory exists
        os.makedirs(output_subdir, exist_ok=True)

        # Get all NIfTI files in the input subdirectory
        nifti_files = [f for f in os.listdir(input_subdir) if f.endswith(('.nii', '.nii.gz'))]

        # Convert each NIfTI file to a numpy array
        for nifti_file in nifti_files:
            input_file = os.path.join(input_subdir, nifti_file)
            output_file = os.path.join(output_subdir, nifti_file.replace('.nii.gz', '.npy').replace('.nii', '.npy'))

            # Load the NIfTI file
            img = nib.load(input_file)

            # Convert the image data to a numpy array and save it
            np.save(output_file, img.get_fdata())

# Use the function
convert_nifti_to_numpy('/home/kaiyu/project/VesselSeg/data/MICCAI_MONAI', '/home/kaiyu/project/VesselSeg/data/SMRA_MONAI_npy')
# convert_nifti_to_numpy('/home/kaiyu/project/VesselSeg/data/CAS2023_test', '/home/kaiyu/project/VesselSeg/data/SMRA_MONAI_test_npy')
