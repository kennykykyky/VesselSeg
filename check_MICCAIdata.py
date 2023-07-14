import glob
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pdb
import numpy as np
import nibabel as nib

path_dataset='./data/CAS2023_training'

save_dir = './tmp/MICCAI_tmp'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
paths_items_all = glob.glob(os.path.join(path_dataset, '*/*.nii.gz'))
paths_items_avail = []

meta_file = pd.read_csv('./data/CAS2023_training/meta.csv')

for path_item in paths_items_all[:]:
    
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
    
    img_data = img.get_fdata()
    seg_data = seg.get_fdata()
    
    ys, xs, zs = np.asarray(img_data > 0.05).nonzero()
    pdb.set_trace()
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
    plt.savefig(os.path.join(save_dir, f'{name_item[:-7]}.png'))

    # Show the plot
    # plt.show()
    
pdb.set_trace()
    