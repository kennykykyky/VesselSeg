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

def generate_json(output_path, filename, num_test, num_validation, num_training, test, validation, training):
    data = {
        "description": "MICCAI CoW Semantic 2023",
        "labels": {
            "background": 0,
            "BA": 1,
            "R-PCA": 2,
            "L-PCA": 3,
            "R-ICA": 4,
            "R-MCA": 5,
            "L-ICA": 6,
            "L-MCA": 7,
            "R-Pcom": 8,
            "L-Pcom": 9,
            "Acom": 10,
            "R-ACA": 11,
            "L-ACA": 12,
            "3rd-A2": 13
        },
        "licence": "miccai",
        "modality": {
            "0": "MRA"
        },
        "name": "MICCAI CAS 2023",
        "numTest": num_test,
        "numValidation": num_validation,
        "numTraining": num_training,
        "reference": "MICCAI CoW 2023",
        "release": "1.0 08/16/2023",
        "tensorImageSize": "3D",
        "test": test,
        "training": training,
        "validation": validation
    }

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(os.path.join(output_path, filename), 'w') as f:
        json.dump(data, f, indent=4)

def generate_data(numbers):
    data = []
    for number in numbers:

        data.append({
            "image": f"imagesTs/{str(number).zfill(3)}.nii.gz",
            "label": f"labelsTs/{str(number).zfill(3)}.nii.gz",
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


path_dataset='./data/CoW_Semantic_ROI'

test_numbers, validation_numbers, training_numbers = split_numbers(1, 110, 10, 20, seed=0)

# train_numbers is the rest of the numbers
training_numbers = [i for i in range(1, 110, 1) if i not in test_numbers and i not in validation_numbers]

test = generate_data(test_numbers)
validation = generate_data(validation_numbers)
training = generate_data(training_numbers)

generate_json('/home/kaiyu/project/VesselSeg/data/CoW_Semantic_ROI', 'CoW_Semantic_ROI.json', len(test), len(validation), len(training), test, validation, training)