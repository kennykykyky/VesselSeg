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
    # Assuming size is a tuple (width, height, depth) and location is a tuple (x, y, z)
    # The center of the bounding box is the location
    center = location
    # The size of the bounding box is the size
    size = size
    # Return as a list of len 6 as integer
    return [int(center[0]), int(center[1]), int(center[2]), int(size[0]), int(size[1]), int(size[2])]

def generate_data(numbers, test = False):
    data = []
    location_list = []
    size_list = []
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
            
            location_list.append(location)
            size_list.append(size)

            # Transform the size and location into a 3d bounding box
            bounding_box = to_3d_bounding_box(size, location)
        
        print(f'bbox size: {size}, bbox location: {location}')

        # visualize_bbox(image_path[0], bounding_box)
    return location_list, size_list

path_dataset='./data/TopCoW_detection'

# train_numbers is the rest of the numbers
numbers = list(range(1, 111))

location_list, size_list = generate_data(numbers, test=True)

# calculate the max value of the first component of the size in all cases
max_size_0 = max([size[0] for size in size_list])
# calculate the max value of the second component of the size in all cases
max_size_1 = max([size[1] for size in size_list])
# calculate the max value of the third component of the size in all cases
max_size_2 = max([size[2] for size in size_list])

print(f'max_size_0: {max_size_0}, max_size_1: {max_size_1}, max_size_2: {max_size_2}')

pdb.set_trace()
