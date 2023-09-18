import glob
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pdb
import numpy as np
import nibabel as nib

# Begin of plotting SMRA-MICCAI data

path_dataset='./data/CAS2023_test'

save_dir = './tmp/MICCAI_test_tmp'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
paths_items_all = glob.glob(os.path.join(path_dataset, '*/*.nii.gz'))
paths_items_avail = []

for path_item in paths_items_all[:]:

    print(path_item)
    
    name_item = os.path.basename(path_item)
    id_item = int(name_item[:-7])
    
    path_img = os.path.join(path_dataset, f'imagesTs/{name_item}')
    
    img = nib.load(path_img)

    # print the shape of the image
    print(img.shape, img.shape[1]/img.shape[2])


    img_data = img.get_fdata()

    mip_x = np.max(img_data, axis=0)
    mip_y = np.max(img_data, axis=1)
    mip_z = np.max(img_data, axis=2)

    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 10))
    # Plot MIP images along x, y, and z axes
    axs[0].imshow(mip_x, cmap='gray')
    axs[0].set_title('MIP along X-axis')
    axs[1].imshow(mip_y, cmap='gray')
    axs[1].set_title('MIP along Y-axis')
    axs[2].imshow(mip_z, cmap='gray')
    axs[2].set_title('MIP along Z-axis')
        
    plt.tight_layout()
    # Save the plot as an image file
    if img.shape[1]/img.shape[2] < 3.5:
        plt.savefig(os.path.join(save_dir, f'{name_item[:-7]}_largeZ.png'))
    else:
        plt.savefig(os.path.join(save_dir, f'{name_item[:-7]}.png'))
    plt.close()

    # plt.show()

# End of plotting SMRA-MICCAI data
    
pdb.set_trace()
    