import json
import shutil
import os
import pdb

# Load the data from the JSON file
with open('/home/kaiyu/project/VesselSeg/data/CoW_bbox_nii/CoW_detection.json', 'r') as f:
    data = json.load(f)

# Get the lists of training, validation, and test cases
training_cases = data.get('training', [])
validation_cases = data.get('validation', [])
test_cases = data.get('test', [])

# Define the source and destination directories for the images
source_dir = '/home/kaiyu/project/VesselSeg/data/TopCoW_detection/image'
dest_dir_tr = '/home/kaiyu/project/VesselSeg/data/CoW_bbox_nii/imagesTr'
dest_dir_ts = '/home/kaiyu/project/VesselSeg/data/CoW_bbox_nii/imagesTs'

pdb.set_trace()

# Copy the training and validation cases to imagesTr
for case in training_cases + validation_cases:
    image_name = case['image'].split('/')[-1]
    # check whether the file already exists in the destination directory
    if not os.path.exists(os.path.join(dest_dir_tr, image_name)):
        shutil.copy(os.path.join(source_dir, image_name), dest_dir_tr)
        # print copy from source_dir/image_name to dest_dir_tr
        print('start copy from {} to {}'.format(os.path.join(source_dir, image_name), dest_dir_tr))

# Copy the test cases to imagesTs
for case in test_cases:
    image_name = case['image'].split('/')[-1]
    if not os.path.exists(os.path.join(dest_dir_ts, image_name)):
        shutil.copy(os.path.join(source_dir, image_name), dest_dir_ts)
        print('start copy from {} to {}'.format(os.path.join(source_dir, image_name), dest_dir_ts))

# copy the label files
source_dir = '/home/kaiyu/project/VesselSeg/save/test_03_11_inferCoW_2023-08-24_15-42-02'
dest_dir_tr = '/home/kaiyu/project/VesselSeg/data/CoW_bbox_nii/labelsTr'
dest_dir_ts = '/home/kaiyu/project/VesselSeg/data/CoW_bbox_nii/labelsTs'

for case in training_cases + validation_cases:
    label_name = case['label'].split('/')[-1]

    # check whether the case exists in source_dir
    if not os.path.exists(os.path.join(source_dir, label_name)):
        print('file {} does not exist in {}'.format(label_name, source_dir))
        continue

    if not os.path.exists(os.path.join(dest_dir_tr, label_name)):
        shutil.copy(os.path.join(source_dir, label_name), dest_dir_tr)
        print('start copy from {} to {}'.format(os.path.join(source_dir, label_name), dest_dir_tr))

for case in test_cases:
    label_name = case['label'].split('/')[-1]
    if not os.path.exists(os.path.join(source_dir, label_name)):
        print('file {} does not exist in {}'.format(label_name, source_dir))
        continue
    if not os.path.exists(os.path.join(dest_dir_ts, label_name)):
        shutil.copy(os.path.join(source_dir, label_name), dest_dir_ts)
        print('start copy from {} to {}'.format(os.path.join(source_dir, label_name), dest_dir_ts))
