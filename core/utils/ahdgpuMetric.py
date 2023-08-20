import numpy as np
import pdb
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from monai.metrics import compute_hausdorff_distance, compute_percent_hausdorff_distance, HausdorffDistanceMetric, compute_average_surface_distance

def ahdgpu_metric(pred_mask, img, gt, spacing, pos_s):

    ahd = compute_average_surface_distance(pred_mask, gt, include_background=False, symmetric=True, spacing = spacing).item()

    nf, nf_s = _compute_norm_factor(img, gt, spacing, pos_s)

    ahd_score = max((nf - ahd) / nf, 0)

    if pos_s.shape[0] == 6:
        mask_s = gt[..., pos_s[0]:pos_s[1], pos_s[2]:pos_s[3], pos_s[4]:pos_s[5]]
        pred_img_s = pred_mask[..., pos_s[0]:pos_s[1], pos_s[2]:pos_s[3], pos_s[4]:pos_s[5]]
        
        if pred_img_s.sum() > 0:
            ahd_s = compute_average_surface_distance(pred_img_s, mask_s, include_background=False, symmetric=True, spacing = spacing).item()
        else:
            ahd_s = nf_s
        
        ahd_score_s = max((nf_s - ahd_s) / nf_s, 0)
    else:
        ahd_score_s = 0

    return ahd_score, ahd_score_s

def _compute_norm_factor(img, mask, spacing, pos_s):
    # compute the norm facter of AHD scores
    # , using AHD scores of segmentation by simple threshold
    ahd = []
    ahd_s = []
    img_max = img.max()
    img_min = None
    for th in [0.2, 0.3, 0.4]:
        if img_max > 0:
            pred_img = img > img_max * th
        else:
            if img_min is None:
                img_min = img.min()
            pred_img = img > img_min * th 
        # if np.any(pred_img) and np.any(mask):   # check if pred_img and mask contain any binary object
        #     ahd.append(assd(pred_img, mask, voxelspacing=spacing))
        # else:
        #     pdb.set_trace()
        ahd.append(compute_average_surface_distance(pred_img, mask, include_background=False, symmetric=True, spacing = spacing).item())

    # determine if pos_s is a shape [6,1] array or length 1 array
    if pos_s.shape[0] == 6:
        mask_s = mask[..., pos_s[0]:pos_s[1], pos_s[2]:pos_s[3], pos_s[4]:pos_s[5]]

        # # plot 2d image of mask[59] as img with the 2d bounding box from img [pos_s[2]:pos_s[3], pos_s[4]:pos_s[5]]
        # mip_x = np.max(img, axis=0)
        # mip_z = np.max(img, axis=2)
        # fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        # # Plot MIP images along x, y, and z axes
        # axs[0, 0].imshow(mip_x, cmap='gray')
        # axs[1, 0].imshow(mip_x, cmap='gray')
        # axs[1, 0].imshow(np.max(mask, axis=0), cmap='jet', alpha=0.4)
        # axs[0, 1].imshow(mip_z, cmap='gray')
        # axs[1, 1].imshow(mip_z, cmap='gray')
        # axs[1, 1].imshow(np.max(mask, axis=2), cmap='jet', alpha=0.4)
        # bbox_coords = pos_s[2:]
        # rect1 = patches.Rectangle((bbox_coords[2], bbox_coords[0]),abs(bbox_coords[3] - bbox_coords[2]),abs(bbox_coords[1] - bbox_coords[0]),linewidth=1, edgecolor = 'red', facecolor='none')
        # axs[1,0].add_patch(rect1)
        # bbox_coords = pos_s[:4]
        # rect2 = patches.Rectangle((bbox_coords[2], bbox_coords[0]),abs(bbox_coords[3] - bbox_coords[2]),abs(bbox_coords[1] - bbox_coords[0]),linewidth=1, edgecolor = 'red', facecolor='none')
        # axs[1,1].add_patch(rect2)
        # plt.savefig('/home/kaiyu/project/VesselSeg/tmp/test_stenosis.png')

        pred_img_s = pred_img[..., pos_s[0]:pos_s[1], pos_s[2]:pos_s[3], pos_s[4]:pos_s[5]]
        if pred_img_s.max() == False:
            pred_img_s[..., int(pred_img_s.shape[0]/2), int(pred_img_s.shape[1]/2), int(pred_img_s.shape[2]/2)] = 1
        # if np.any(pred_img_s) and np.any(mask_s):   # check if pred_img and mask contain any binary object
        #     ahd_s.append(assd(pred_img_s, mask_s, voxelspacing=spacing))
        # else:
        #     ahd_s.append(ahd_s.mean())
        ahd_s.append(compute_average_surface_distance(pred_img_s, mask_s, include_background=False, symmetric=True, spacing = spacing).item())
    else:
        ahd_s.append(-1)

    nf = np.mean(ahd)
    nf_s = np.mean(ahd_s) if ahd_s[0] != -1 else -1
    return nf, nf_s