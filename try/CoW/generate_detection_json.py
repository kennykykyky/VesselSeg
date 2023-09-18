import glob
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pdb
import random
import numpy as np
import nibabel as nib
import json

# Try to add stenosis location & spacing info to the MONAI SMRA dataset
# The stenosis information and spacing info will be inserted under key 'extra'

def visualize_bbox(image_path, bbox):
    # Read the image
    image = nib.load(image_path).get_fdata()
    # Get the shape of the image
    shape = image.shape

    # select five slices in axial view evenly distributed in the middle slab of the image
    slices = np.linspace(bbox[2], bbox[2] + bbox[-1], 5, dtype=int)

    # create a figure with five subplots, for each it also includes two images
    fig, axs = plt.subplots(5, 2, figsize=(10, 10))
    # plot the five slices: on the left is the original image slice and the bbox on it, the right is the image slice with bbox cropped
    for i, slice in enumerate(slices):
        # plot the original image slice
        axs[i, 0].imshow(image[:, :, slice], cmap='gray')
        # plot the bbox on the original image slice
        axs[i, 0].add_patch(patches.Rectangle((bbox[1], bbox[0]), bbox[4], bbox[3], linewidth=1, edgecolor='r', facecolor='none'))
        # plot the cropped image slice
        axs[i, 1].imshow(image[int(bbox[0]):int(bbox[0]+bbox[3]), int(bbox[1]):int(bbox[1]+bbox[4]), slice], cmap='gray')

    # save the figure to ./tmp/Cow
    if not os.path.exists('./tmp/Cow'):
        os.makedirs('./tmp/Cow')
    plt.savefig('./tmp/Cow/' + image_path.split('/')[-1].split('.')[0] + '.png')
    plt.close()

def to_3d_bounding_box(size, location):
    center = location
    return [int(center[0]), int(center[1]), int(center[2]), int(center[0]+size[0]), int(center[1]+size[1]), int(center[2] + size[2])]

def generate_json(output_path, filename, num_test, num_validation, num_training, test, validation, training):
    data = {
        "description": "MICCAI CoW 2023",
        "labels": {
            "0": "background",
            "1": "artery"
        },
        "licence": "miccai",
        "modality": {
            "0": "MRA"
        },
        "name": "MICCAI CoW 2023",
        "numTest": num_test,
        "numValidation": num_validation,
        "numTraining": num_training,
        "reference": "MICCAI CoW 2023",
        "release": "1.0 08/24/2023",
        "tensorImageSize": "3D",
        "test": test,
        "training": training,
        "validation": validation
    }

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(os.path.join(output_path, filename), 'w') as f:
        json.dump(data, f, indent=4)

def generate_data(numbers, test = False):
    data = []
    for number in numbers:

        print(f'Processing {number}...')

        # search for files under /image and /bbox with the same number
        image_path = glob.glob(os.path.join(path_dataset, 'image', f'{str(number).zfill(3)}*'))
        bbox_path = glob.glob(os.path.join(path_dataset, 'bbox', f'{str(number).zfill(3)}*.txt'))

        # read the .txt bbox file and convert it to a list
        if len(bbox_path) == 1:
            with open(bbox_path[0], 'r') as f:
                lines = f.readlines()

            # For each line in the file
            for i, line in enumerate(lines[1:]):
                # Split the line into size and location
                if i == 0:
                    size_str = line.strip().split(': ')[-1]
                    size = tuple(map(float, size_str.split()))
                else:
                    location_str = line.strip().split(': ')[-1]
                    location = tuple(map(float, location_str.split()))

            # Transform the size and location into a 3d bounding box
            bounding_box = to_3d_bounding_box(size, location)

        # visualize_bbox(image_path[0], bounding_box)

        if not test:
            data.append({
                "image": f"imagesTr/{str(number).zfill(3)}.nii.gz",
                "seg": f"labelsTr/{str(number).zfill(3)}.nii.gz",
                "label": [0],
                "box": [bounding_box]
            })
        else:
            data.append({
                "image": f"imagesTs/{str(number).zfill(3)}.nii.gz",
                "seg": f"labelsTs/{str(number).zfill(3)}.nii.gz",
                "label": [0],
                "box": [bounding_box]
            })
    return data

def split_numbers(start, end, test_len, validation_len, seed=0):

    # Set the random seed
    random.seed(seed)

    # Create a list of all numbers
    numbers = list(range(start, end+1))

    # Randomly shuffle the numbers
    random.shuffle(numbers)

    # Split the numbers into test, validation, and training groups
    test = numbers[:test_len]
    validation = numbers[test_len:test_len+validation_len]
    training = numbers[test_len+validation_len:]

    return test, validation, training


path_dataset='./data/TopCoW_detection'

test_numbers, validation_numbers, training_numbers = split_numbers(1, 110, 10, 20, seed=0)

# train_numbers is the rest of the numbers
training_numbers = [i for i in range(1, 110, 1) if i not in test_numbers and i not in validation_numbers]

test = generate_data(test_numbers, test=True)
validation = generate_data(validation_numbers)
training = generate_data(training_numbers)

generate_json('/home/kaiyu/project/VesselSeg/data/CoW_bbox_nii', 'CoW_detection.json', len(test), len(validation), len(training), test, validation, training)