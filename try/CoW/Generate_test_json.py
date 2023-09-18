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

def generate_json(output_path, filename, num_test, test):
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
        "name": "MICCAI CAS 2023",
        "numTest": num_test,
        "numValidation": 0,
        "numTraining": 0,
        "reference": "MICCAI CoW 2023",
        "release": "1.0 08/16/2023",
        "tensorImageSize": "3D",
        "test": test
    }

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(os.path.join(output_path, filename), 'w') as f:
        json.dump(data, f, indent=4)

def generate_data(numbers):
    data = []
    for number in numbers:

        data.append({
            "image": f"imagesTs/{str(number).zfill(3)}.nii.gz"
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


path_dataset='./data/CAS2023_test'

# create test numbers from 0 to 49
test_numbers = list(range(1, 111))
test = generate_data(test_numbers)

generate_json('/home/kaiyu/project/VesselSeg/data/CoW_bbox_nii', 'CoW_wholebrain_artseg.json', len(test), test)