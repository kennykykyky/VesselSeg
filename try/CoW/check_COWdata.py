import glob
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pdb
import numpy as np
import nibabel as nib
import plotly.graph_objects as go

import gudhi as gd

# Function to create 3D donut shape
def make_torus(r, tube_radius, grid_size):
    z,y,x = np.indices((grid_size,grid_size,grid_size))
    z = z - grid_size//2
    y = y - grid_size//2
    x = x - grid_size//2
    outer = (x**2 + y**2 + z**2 + r**2 - tube_radius**2)**2 
    inner = 4*r**2 * (x**2 + y**2)
    mask = inner - outer <= 0
    return mask

# Test this function with some parameters:
torus = make_torus(30, 10, 100)

# Create a meshgrid for the x, y, and z coordinates
x, y, z = np.meshgrid(np.arange(torus.shape[0]),
                    np.arange(torus.shape[1]),
                    np.arange(torus.shape[2]))

fig = go.Figure(data=go.Volume(
    x=x.flatten(),
    y=y.flatten(),
    z=z.flatten(),
    value=torus.flatten(),
    isomin=0.1,
    isomax=0.8,
    opacity=0.1,  # needs to be small to see through all surfaces
    surface_count=20,  # needs to be a large number for good volume rendering
    ))

fig.show()

# Create a cubical complex from the 3D image
cc = gd.CubicalComplex(dimensions=torus.shape, top_dimensional_cells=torus.flatten())

# Compute persistence diagram
diag = cc.persistence()

# Compute betti numbers
betti_numbers = cc.betti_numbers()
print(betti_numbers)
pdb.set_trace()


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
    
def rename_COWdata(datapath, name='TOF', label = False):
    
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
datapath = 'D:/Kaiyu/nnUNet_dataset/nnUNet_raw/Dataset302_COWMRA/labelsTr'
# rename_nnUnetdata(datapath, name='TOF', label = True)
# rename_COWdata(datapath, name='COWMRROI', label = True)

pdb.set_trace()

label_file = r"D:\Kaiyu\nnUNet_dataset\nnUNet_raw\Dataset302_COWMRA\labelsTr"
label_files = glob.glob(os.path.join(label_file, '*.nii.gz'))

for path_item in label_files:
    seg = nib.load(path_item)
    seg_data = seg.get_fdata()
    # print(seg_data.max())
    old_data = seg_data.copy()
    
    print(path_item.split('\\')[-1])
    # my_list = list(range(13)) + [15]
    # print(list(set(my_list) - set(np.unique(seg_data))))
    
    # replace the pixels with value 15 to 13
    seg_data[seg_data == 15] = 13
    print(np.unique(seg_data))
    
    # overwrite the original segmentation mask in nii format, the data should be integer
    seg_new = nib.Nifti1Image(seg_data.astype(np.uint8), seg.affine, seg.header)
    
    # save the nii file
    nib.save(seg_new, path_item)
    
    # test = nib.load(path_item.replace('001', '001_test'))
    # print(np.unique(test.get_fdata()))
    # pdb.set_trace()


pdb.set_trace()

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
    