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
        "description": "MICCAI CAS 2023",
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
        "numValidation": num_validation,
        "numTraining": num_training,
        "reference": "MICCAI CAS 2023",
        "release": "1.0 06/08/2023",
        "tensorImageSize": "3D",
        "test": test,
        "training": training,
        "validation": validation
    }

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(os.path.join(output_path, filename), 'w') as f:
        json.dump(data, f, indent=4)

def generate_data(numbers, df, test = False):
    data = []
    for number in numbers:
        pos_s = df.loc[df['index'] == number, 'pos_s'].values[0]
        spacing = df.loc[df['index'] == number, 'spacing'].values[0]

        pos_s = pos_s.strip('[]')
        bbox_list = pos_s.split(',')
        pos_s = np.array(bbox_list, dtype=int)
        spacing = np.array([float(i) for i in spacing[1:-1].split(',')])

        # move the last element of spacing to the first, and move the last two elements of pos_s to the first two
        # roll the spacing array by 1 to the right
        spacing = np.roll(spacing, 1)

        # roll the pos_s array by 2 to the right
        pos_s = np.roll(pos_s, 2)

        # Align the format for pos_s and spacing in the json file, where there is just one space between each number
        pos_s = '[' + ' '.join([str(i) for i in pos_s]) + ']'
        spacing = '[' + ' '.join([str(i) for i in spacing]) + ']'

        if not test:
            data.append({
                "image": f"imagesTr/{str(number).zfill(3)}.nii.gz",
                "label": f"labelsTr/{str(number).zfill(3)}.nii.gz",
                "extra": f"{pos_s},{spacing}"
            })
        else:
            data.append({
                "image": f"imagesTs/{str(number).zfill(3)}.nii.gz",
                "label": f"labelsTs/{str(number).zfill(3)}.nii.gz",
                "extra": f"{pos_s},{spacing}"
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


path_dataset='./data/CAS2023_training'

meta_df = pd.read_csv('./data/CAS2023_training/meta_with_nf.csv')

# test_numbers, validation_numbers, training_numbers = split_numbers(0, 99, 10, 20, seed=0)
test_numbers = [0, 16, 25, 41, 60, 67, 72, 78, 85, 91]
validation_numbers = [77, 79, 80, 81, 82, 83, 84, 86, 87, 88, 89, 90, 92, 93, 94, 95, 96, 97, 98, 99]
# train_numbers is the rest of the numbers
training_numbers = [i for i in range(100) if i not in test_numbers and i not in validation_numbers]

test = generate_data(test_numbers, meta_df, test=True)
validation = generate_data(validation_numbers, meta_df)
training = generate_data(training_numbers, meta_df)

generate_json('/home/kaiyu/project/VesselSeg/data/MICCAI_MONAI', 'MICCAI_CAS_2023_extra.json', len(test), len(validation), len(training), test, validation, training)