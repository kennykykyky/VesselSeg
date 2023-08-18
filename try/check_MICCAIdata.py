import glob
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pdb
import numpy as np
import nibabel as nib

# Begin of renaming nnUnet data

def rename_nnUnetdata(datapath, name='TOF', label = False):
    
    # rename the data inside datapath to '<name>_xxx_0000.nii.gz', where xxx is the id number for each case and it starts from 001 to the total number of cases
    # e.g. rename_nnUnetdata('./data/CAS2020_training', name='TOF')
    
    paths_items = glob.glob(os.path.join(datapath, '*.nii.gz'))
    paths_items.sort()
    for i, path_item in enumerate(paths_items):
        name_item = os.path.basename(path_item)
        if not label:
            name_item_new = f'{name}_{i+1:03d}_0000.nii.gz'
        else:
            name_item_new = f'{name}_{i+1:03d}.nii.gz'
        os.rename(path_item, os.path.join(os.path.dirname(path_item), name_item_new))
        print(f'{name_item} -> {name_item_new}')
        
# rename_nnUnetdata('D:\Kaiyu\nnUNet_dataset\nnUNet_raw\Dataset301_SMRATOF\imagesTr', name='TOF')
# datapath = 'D:/Kaiyu/nnUNet_dataset/nnUNet_raw/Dataset301_SMRATOF/labelsTr'
# rename_nnUnetdata(datapath, name='TOF', label = True)

# pdb.set_trace()

# End of renaming nnUnet data


# Begin of plotting SMRA-MICCAI data

path_dataset='./data/CAS2023_training'

save_dir = './tmp/MICCAI_tmp'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
paths_items_all = glob.glob(os.path.join(path_dataset, '*/*.nii.gz'))
paths_items_avail = []

meta_file = pd.read_csv('./data/CAS2023_training/meta.csv')

for path_item in paths_items_all[:]:

    print(path_item)
    
    name_item = os.path.basename(path_item)
    id_item = int(name_item[:-7])
    
    pos_s = meta_file.loc[meta_file['index'] == id_item, 'pos_s'].values[0]
    pos_s = pos_s.strip('[]')
    bbox_list = pos_s.split(',')
    pos_s = np.array(bbox_list, dtype=int)
    
    path_img = os.path.join(path_dataset, f'data/{name_item}')
    path_seg = os.path.join(path_dataset, f'mask/{name_item}')
    
    img = nib.load(path_img)
    seg = nib.load(path_seg)

    # print the shape of the image
    print(img.shape, img.shape[1]/img.shape[2])

    img_data = img.get_fdata()
    seg_data = seg.get_fdata()
    
    ys, xs, zs = np.asarray(img_data > 0.05).nonzero()
    # pos_s = [img_data.shape[0] - pos_s[0], img_data.shape[0] - pos_s[1], img_data.shape[1] - pos_s[2], img_data.shape[1] - pos_s[3], img_data.shape[2] - pos_s[4], img_data.shape[2] - pos_s[5]]
    
    # Create the maximum intensity projection of the image
    mip_x = np.max(img_data, axis=0)
    mip_y = np.max(img_data, axis=1)
    mip_z = np.max(img_data, axis=2)

    # Create subplots
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    # Plot MIP images along x, y, and z axes
    axs[0, 0].imshow(mip_x, cmap='gray')
    axs[0, 0].set_title('MIP along X-axis')
    axs[0, 1].imshow(mip_y, cmap='gray')
    axs[0, 1].set_title('MIP along Y-axis')
    axs[0, 2].imshow(mip_z, cmap='gray')
    axs[0, 2].set_title('MIP along Z-axis')

    # Plot segmentation mask images along x, y, and z axes
    axs[1, 0].imshow(mip_x, cmap='gray')
    axs[1, 0].imshow(np.max(seg_data, axis=0), cmap='jet', alpha=0.4)
    axs[1, 0].set_title('Segmentation Mask along X-axis')
    axs[1, 1].imshow(mip_y, cmap='gray')
    axs[1, 1].imshow(np.max(seg_data, axis=1), cmap='jet', alpha=0.4)
    axs[1, 1].set_title('Segmentation Mask along Y-axis')
    axs[1, 2].imshow(mip_z, cmap='gray')
    axs[1, 2].imshow(np.max(seg_data, axis=2), cmap='jet', alpha=0.4)
    axs[1, 2].set_title('Segmentation Mask along Z-axis')
    
    if not pos_s.shape[0] == 1:

        # Add bounding boxes to the MIP images
        for i, ax in enumerate(axs.flatten()):
            if i < 3:
                if i == 0:
                    bbox_coords = pos_s[2:]
                elif i == 1:
                    bbox_coords = np.concatenate((pos_s[:2], pos_s[4:]))
                else:
                    bbox_coords = pos_s[:4]
                    
                # (row, col) for numpy but (col, row) for matplotlib
                rect = patches.Rectangle((bbox_coords[2], bbox_coords[0]),
                                            abs(bbox_coords[3] - bbox_coords[2]),
                                            abs(bbox_coords[1] - bbox_coords[0]),
                                            linewidth=1, edgecolor = 'red', facecolor='none')
                
                rect2 = patches.Rectangle((bbox_coords[2], bbox_coords[0]),
                                            abs(bbox_coords[3] - bbox_coords[2]),
                                            abs(bbox_coords[1] - bbox_coords[0]),
                                            linewidth=1, edgecolor = 'red', facecolor='none')
                ax.add_patch(rect)
                axs.flatten()[i+3].add_patch(rect2)            

            ax.axis('off')
        
    plt.tight_layout()
    # Save the plot as an image file
    if img.shape[1]/img.shape[2] < 3.5:
        plt.savefig(os.path.join(save_dir, f'{name_item[:-7]}_largeZ.png'))
    else:
        plt.savefig(os.path.join(save_dir, f'{name_item[:-7]}.png'))
    plt.close()

    # Show the plot
    # plt.show()

# End of plotting SMRA-MICCAI data
    
pdb.set_trace()
    