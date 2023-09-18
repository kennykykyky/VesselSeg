from core.networks.unet import R2U_Net, UNet, NestedUNet
from core.networks.unet3d import Unet3D
# from core.utils.img import get_img_grid, draw_conts_given_mask
from core.losses.DiceLoss import Dice
from core.losses.focal import BinaryFocalLossWithLogits
from core.losses.cldiceLoss import soft_cldice, soft_dice_cldice, soft_dice
from core.losses.LumenLoss import LumenLoss, Binary
from core.utils.clDiceMetric import clDice
from core.utils.bettiMetric import betti_error_metric
from core.utils.ahdMetric import ahd_metric
from core.utils.ahdgpuMetric import ahdgpu_metric
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager

from monai.data import (
    decollate_batch,
)
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, compute_hausdorff_distance
from monai.transforms import AsDiscrete
from monai.losses import DiceLoss, FocalLoss
from monai.metrics import compute_hausdorff_distance, compute_percent_hausdorff_distance, HausdorffDistanceMetric, compute_average_surface_distance

import time
import torch
import pdb
import cv2
from skimage.filters import frangi
from torchvision.utils import make_grid
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from einops.einops import rearrange, repeat
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from skimage import measure, morphology
from collections import OrderedDict
from sklearn import metrics
from collections import OrderedDict
import torch.distributed as dist
import os
import json
import pandas as pd
import nibabel as nib
from scipy.ndimage import label, binary_fill_holes, find_objects
from skimage.measure import regionprops
from skimage.segmentation import flood_fill

mpl.use('agg')

class nnUNet(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
    def forward(self, x):
        x = self.net(x)
        return x[0]

class nnUNetModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # load configuration and dataset json files from pretrained nnUNet model
        with open(os.path.join(cfg.exp.nnunet_result.path, 'dataset.json'), 'r') as f:
            dataset_json = json.load(f)
        plans_manager = PlansManager(os.path.join(cfg.exp.nnunet_result.path, 'plans.json'))
        with open(os.path.join(cfg.exp.nnunet_result.path, 'plans.json'), 'r') as f:
                self.plans = json.load(f)

        # load pretrained nnUNet model
        self.net = get_network_from_plans(plans_manager, 
                                          dataset_json, 
                                          plans_manager.get_configuration(cfg.exp.nnunet_result.model), 
                                          num_input_channels=1)

        # load pretrained weights
        id_device = self.cfg.exp.idx_device
        map_location = {'cuda:0': f'cuda:{id_device}'}
        saved_model = torch.load(cfg.exp.nnunet_result.weights, map_location=map_location)
        pretrained_dict = saved_model['network_weights']
        self.net.load_state_dict(pretrained_dict, strict=True)
        # Since the output of nnUNet include results from all decoder layers, we only need the last one
        # Therefore we wrap the nnUNet model with a new model to extract the last decoder layer output
        self.net = nnUNet(self.net)

        self.cfg.var.obj_model = self
        self.paths_file_net = []

        # Here we import the different losses
        self.focal_loss = BinaryFocalLossWithLogits(alpha=cfg.model.w_focal, reduction='mean') # already has sigmoid
        self.cldice_loss = soft_dice_cldice(alpha=self.cfg.model.w_cl)
        self.lumen_loss = LumenLoss(sample_weight = None, pos_weight =1.00, w_ce =self.cfg.model.w_ce, w_dt = self.cfg.model.w_dt, w_ace=self.cfg.model.w_ace, w_dice=self.cfg.model.w_dice)

        # Here we import the different metrics
        self.dice = Dice()
        self.binary = Binary(th = 0.5, gamma=20)
        self.threshold = 0.9 # threshold for binary segmentation

        self.gt2onehot = AsDiscrete(to_onehot=2)
        self.pred2onehot = AsDiscrete(to_onehot=2, argmax=True)

        # includes each metric and the total loss for backward training
        self.metrics_iter: dict[str, torch.Tensor | float] = {}
        self.metrics_epoch: dict[str, torch.Tensor | float] = {}
        # [B, M, ...]
        self.imgs: torch.Tensor = None
        self.output: dict[str, torch.Tensor] = {}
        # [B, M, ...], ground truth segmentation
        self.output['gt']: torch.Tensor = None
        # [B, 1, ...], predicted segmentation, binary float 
        self.output['mask']: torch.Tensor = None

    def forward(self, data):
        # should consider whether use single channel or one-hot for segmentation as the input for loss function
        # We will first transform labels into one-hot format, but select only second channel for binary segmentation to get more stable numerical results

        if 'label' in data.keys():
            imgs, segs = data['image'], data['label'].long()
            # for monai dataset, we usually don't use GPU to accelerate data loading and therefore we will move data to GPU here
            imgs, segs = imgs.to(self.cfg.var.obj_operator.device), segs.to(self.cfg.var.obj_operator.device)
            # The gt2onehot function will transform the segmentation into one-hot format
            # The input of the gt2onehot function should be [C, B, ...], where C is the number of classes
            self.output['gt'] = self.gt2onehot(segs.permute(1, 0, 2, 3, 4)).permute(1, 0, 2, 3, 4)
        else:
            imgs = data['image']
            imgs = imgs.to(self.cfg.var.obj_operator.device)
            segs = None

        self.imgs = imgs

        # check if 'weight' is in data
        if 'weight' in data.keys():
            self.sample_weight = data['weight'].view(-1,1).to(self.cfg.var.obj_operator.device)
        else:
            self.sample_weight = None

        self.cfg.var.n_samples = b = imgs.shape[0]
        
        if not self.training:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    # sliding window inference is used to get better inference results (from MONAI)
                    self.output['logits'] = sliding_window_inference(self.imgs, roi_size=self.plans['configurations'][self.cfg.exp.nnunet_result.model]['patch_size'], sw_batch_size=4, predictor=self.net)
                    if 'meta' in data.keys():
                        self.output['spacing'] = data['meta']['spacing'].detach().cpu().numpy().reshape(3,-1).squeeze()
                        self.output['s_bbox'] = data['meta']['s_bbox'].detach().cpu().numpy().astype(np.int32)
                        self.output['s_bbox'] = self.output['s_bbox'].reshape(-1,1) if self.output['s_bbox'].shape[-1] != 6 else self.output['s_bbox'].reshape(6,-1).squeeze()
                    self.output['image_meta_dict'] = data['image_meta_dict']
        else:
            with torch.cuda.amp.autocast():
                self.output['logits'] = self.net(self.imgs)

        # for binary segmentation, the binary function can generate a better segmentation sensitivity
        # self.output['mask'] = (self.binary(torch.sigmoid(self.output['logits'][:,1])) > self.threshold).float() # single channel mask
        self.output['mask'] = torch.argmax(self.output['logits'], dim=1) # single channel mask

        # transform single channel mask into one-hot format and save as self.output['mask_onehot']
        self.output['mask_onehot'] = self.gt2onehot(self.output['mask'].unsqueeze(0)).permute(1, 0, 2, 3, 4)

        del data, segs, imgs

        return self.output['logits'][:,1] # actually this return value is not important if you don't use it in get_metrics() function
    
    def loss_focal(self):
        loss = self.focal_loss(self.output['logits'][:, 1], self.output['gt'][:, 1])
        return loss
    
    def loss_cldiceloss(self):
        # the input are in one-hot format but will be transformed into single channel format inside the clDice function
        loss = self.cldice_loss(self.output['gt'], self.output['logits'])
        return loss

    def loss_lumen(self):
        # The input for loss function should be [B, C, ...], where C is 2 for binary segmentation
        # loss = self.lumen_loss(self.imgs, self.output['logits'], self.output['gt'][:, 1], self.sample_weight)
        loss = self.lumen_loss(self.imgs, self.output['logits'], self.output['gt'][:, 1].unsqueeze(1), self.sample_weight)
        return loss

    def before_epoch(self, mode='train', i_repeat=0):
        self.metrics_epoch = OrderedDict()

        if self.cfg.exp.mode == 'test':
            if self.cfg.exp.test.classification_curve.enable:
                self.gt_roc = []
                self.pred_roc = []

    def after_epoch(self, mode='train'):

        if isinstance(self.output['gt'], torch.Tensor):

            for k, v in self.metrics_epoch.items():
                if self.training:
                    self.metrics_epoch[k] = v / len(getattr(self.cfg.var.obj_operator, f'{mode}_set')) / self.cfg.dataset.num_samples
                else:
                    self.metrics_epoch[k] = v / len(getattr(self.cfg.var.obj_operator, f'{mode}_set'))
                
            if not self.training:
                self.metrics_epoch['metric_final'] = self.metrics_epoch['final_score']

        # if self.cfg.exp.mode == 'test':
        #     if self.cfg.exp.test.classification_curve.enable:
        #         seg_gt_roc = np.concatenate(self.gt_roc)
        #         seg_pred_roc = np.concatenate(self.pred_roc)
        #         fpr, tpr, thres = metrics.roc_curve(y_true=seg_gt_roc.reshape(-1), y_score=seg_pred_roc.reshape(-1))
        #         df = pd.DataFrame({'fp_rate': fpr, 'tp_rate': tpr, 'threshold': thres})
        #         df.to_csv(os.path.join(self.cfg.var.obj_operator.path_exp, 'roc_curve_pointwise.csv'), index=False)
        #         precision, recall, thres = metrics.precision_recall_curve(y_true=seg_gt_roc.reshape(-1),
        #                                                                   probas_pred=seg_pred_roc.reshape(-1))
        #         df = pd.DataFrame({
        #             'precision': precision,
        #             'recall': recall,
        #             'threshold': np.concatenate([thres, [99999]])
        #         })
        #         df.to_csv(os.path.join(self.cfg.var.obj_operator.path_exp, 'precision_recall_curve_pointwise.csv'),
        #                   index=False)

        #         class_gt = np.max(np.max(seg_gt_roc, axis=2), axis=2)[:, 0] # [B]
        #         class_pred = np.max(np.max(seg_pred_roc, axis=2), axis=2)[:, 0] # [B]
        #         fpr, tpr, thres = metrics.roc_curve(y_true=class_gt, y_score=class_pred)
        #         df = pd.DataFrame({'fp_rate': fpr, 'tp_rate': tpr, 'threshold': thres})
        #         df.to_csv(os.path.join(self.cfg.var.obj_operator.path_exp, 'roc_curve_slicewise.csv'), index=False)
        #         precision, recall, thres = metrics.precision_recall_curve(y_true=class_gt, probas_pred=class_pred)
        #         df = pd.DataFrame({
        #             'precision': precision,
        #             'recall': recall,
        #             'threshold': np.concatenate([thres, [99999]])
        #         })
        #         df.to_csv(os.path.join(self.cfg.var.obj_operator.path_exp, 'precision_recall_curve_slicewise.csv'),
        #                   index=False)

        #         dices = []
        #         thresholds = np.linspace(np.min(seg_pred_roc), np.max(seg_pred_roc), 80)
        #         for thre in thresholds:
        #             seg_pred_roc_thres = (seg_pred_roc > thre).astype(int)
        #             numerator = 2 * np.sum(seg_gt_roc * seg_pred_roc_thres, axis=tuple(range(len(seg_gt_roc.shape))))
        #             denominator = np.sum(seg_gt_roc + seg_pred_roc_thres, axis=tuple(range(len(seg_gt_roc.shape))))
        #             dice = (numerator + 1e-8) / (denominator + 1e-8)
        #             dices.append(dice)
        #         df = pd.DataFrame({'dice': dices, 'threshold': thresholds})
        #         df.to_csv(os.path.join(self.cfg.var.obj_operator.path_exp, 'thres_dice_curve.csv'), index=False)

    def get_metrics(self, data, output, mode='train'):
        """
            loss_final: used for backward training
            metric_final: used to select best model (higher is better)
            other metrics: for visualization
        """
        self.metrics_iter = OrderedDict(loss_final=0.)

        if isinstance(self.output['gt'], torch.Tensor):
            for name_loss, w in self.cfg.model.ws_loss.items():
                if w > 0.:
                    loss = getattr(self, f'loss_{name_loss}')()
                    self.metrics_iter[name_loss] = loss.item()
                    self.metrics_iter['loss_final'] += w * loss

            with torch.no_grad():
                
                    seg_gt = self.output['gt'][:,1].type(torch.int64)
                    dice = self.dice(seg_gt, self.output['mask'], dims_sum=tuple(range(1, len(seg_gt.shape))), return_before_divide=False)

                    self.metrics_iter['dice'] = dice.mean().item()
                    self.metrics_iter['clDice'] = clDice(self.output['mask'].detach().cpu().numpy().squeeze().astype(np.uint8), seg_gt.detach().cpu().numpy().squeeze().astype(np.uint8))

                    if mode in ['val', 'test']:

                        self.metrics_iter['ahd'], self.metrics_iter['ahd_s'] = ahdgpu_metric(self.output['mask_onehot'], self.imgs, self.output['gt'], self.output['spacing'], self.output['s_bbox'])

                        # self.metrics_iter['ahd'], self.metrics_iter['ahd_s'] = ahd_metric(self.output['mask'].detach().cpu().numpy().squeeze(),
                        #                                     self.imgs.detach().cpu().numpy().squeeze(),
                        #                                     self.output['gt'][:,1].detach().cpu().numpy().squeeze(),
                        #                                     self.output['spacing'],
                        #                                     self.output['s_bbox'])

                        pos_s = self.output['s_bbox']
                        if pos_s.shape[0] == 6:
                            self.metrics_iter['dice_s'] = self.dice(seg_gt[:, pos_s[0]:pos_s[1], pos_s[2]:pos_s[3], pos_s[4]:pos_s[5]], 
                                                                self.output['mask'][:,pos_s[0]:pos_s[1], pos_s[2]:pos_s[3], pos_s[4]:pos_s[5]], 
                                                                dims_sum=tuple(range(len(seg_gt.shape))), 
                                                                return_before_divide=False).detach().cpu().numpy().squeeze()
                            self.metrics_iter['final_score'] = 0.35 * self.metrics_iter['dice'] + 0.35 * self.metrics_iter['ahd']+ 0.15 * self.metrics_iter['ahd_s']+ 0.15 * self.metrics_iter['dice_s']
                        else:
                            self.metrics_iter['dice_s'] = 0
                            self.metrics_iter['final_score'] = 0.5 * self.metrics_iter['dice'] + 0.5 * self.metrics_iter['ahd']
                        
            for k, v in self.metrics_iter.items():
                self.metrics_epoch[k] = self.metrics_epoch.get(k, 0.) + float(v) * self.imgs.shape[0]

        if self.cfg.exp.mode == 'test':

            # filter the vessels around outside ICA at the first several z-slices
            # if self.imgs.shape[-1]/self.imgs.shape[-3] < 3.5:
            #     case_id = self.output['image_meta_dict']['filename_or_obj'][0].split('/')[-1].split('.')[0]
            #     if case_id not in ['013', '020', '029', '031']:
            #         self.output['mask'] = self.filter_vessel(self.output['mask']).to(torch.int64)

            if self.cfg.exp.test.classification_curve.enable:
                self.gt_roc.append(seg_gt.cpu().numpy())
                self.pred_roc.append(self.output['logits'].cpu().numpy())
            
            check = True
            if check:
                # plot the Maximum intensity projection in 3 directions of the image and the mask, where there are 6 plots that range in 2*3 grid
                # the first row is the image and the second row is the mask
                # the first column is the axial view, the second column is the coronal view, and the third column is the sagittal view
                fig, ax = plt.subplots(2, 3, figsize=(15, 10))
                # plot the axial view
                ax[0, 0].imshow(np.max(self.imgs.detach().cpu().numpy().squeeze(), axis=0), cmap='gray')
                ax[1, 0].imshow(np.max(self.output['mask'].detach().cpu().numpy().squeeze(), axis=0), cmap='gray')
                # plot the coronal view
                ax[0, 1].imshow(np.max(self.imgs.detach().cpu().numpy().squeeze(), axis=1), cmap='gray')
                ax[1, 1].imshow(np.max(self.output['mask'].detach().cpu().numpy().squeeze(), axis=1), cmap='gray')
                # plot the sagittal view
                ax[0, 2].imshow(np.max(self.imgs.detach().cpu().numpy().squeeze(), axis=2), cmap='gray')
                ax[1, 2].imshow(np.max(self.output['mask'].detach().cpu().numpy().squeeze(), axis=2), cmap='gray')
                # save the figure
                plt.savefig(os.path.join(self.cfg.var.obj_operator.path_exp, self.output['image_meta_dict']['filename_or_obj'][0].split('/')[-1].split('.')[0] + '.png'))
                plt.close()

                if isinstance(self.output['gt'], torch.Tensor):
                    # Save 3d difference map between ground truth and predicted masks
                    # Compute differences
                    FP = np.logical_and(self.output['mask'].detach().cpu().numpy().squeeze() == 1, self.output['gt'][0, 1].detach().cpu().numpy().squeeze() == 0)
                    FN = np.logical_and(self.output['mask'].detach().cpu().numpy().squeeze() == 0, self.output['gt'][0, 1].detach().cpu().numpy().squeeze() == 1)
                    TP = np.logical_and(self.output['mask'].detach().cpu().numpy().squeeze() == 1, self.output['gt'][0, 1].detach().cpu().numpy().squeeze() == 1)

                    # Assign unique intensity values for visualization
                    difference_data = np.zeros_like(self.output['gt'][0, 1].detach().cpu().numpy().squeeze())
                    difference_data[FP] = 1  # False Positives marked as 1, red
                    difference_data[FN] = 2  # False Negatives marked as 2, green
                    difference_data[TP] = 3  # True Positives marked as 3, blue
                    # Note: True Negatives will have a value of 0
                    difference_data = np.transpose(difference_data, (1, 2, 0))

                    # Save the difference as a new NIfTI file
                    difference_nii = nib.Nifti1Image(difference_data, np.eye(4))
                    nib.save(difference_nii, os.path.join(self.cfg.var.obj_operator.path_exp, self.output['image_meta_dict']['filename_or_obj'][0].split('/')[-1].split('.')[0] + '_diff.nii.gz'))

            if self.cfg.exp.test.save_seg.enable:
                # save image as nii file
                seg = self.output['mask'].detach().cpu().numpy().squeeze().astype(np.uint8)
                baseid = self.output['image_meta_dict']['filename_or_obj'][0].split('/')[-1].split('.')[0]
                seg = np.transpose(seg, (1, 2, 0))

                # flip the last axis of seg only for CoW data
                if self.cfg.exp.test.save_seg.flip:
                    seg = np.flip(seg, axis=2)

                seg = nib.Nifti1Image(seg, np.eye(4))
                # save the seg file with the same name as in the image_meta_dict
                nib.save(seg, os.path.join(self.cfg.var.obj_operator.path_exp, baseid + '.nii.gz'))
            
            # save the metrics in text file
            with open(os.path.join(self.cfg.var.obj_operator.path_exp, 'metrics.txt'), 'a') as f:
                f.write(self.output['image_meta_dict']['filename_or_obj'][0].split('/')[-1].split('.')[0] + '\n')
                f.write(str(self.metrics_iter) + '\n')                
            
            # print the case name
            print(self.output['image_meta_dict']['filename_or_obj'][0].split('/')[-1].split('.')[0])
            # print the items in the self.metrics_iter
            for k, v in self.metrics_iter.items():
                print(k, v)

        return self.metrics_iter

    def vis(self, writer, global_step, data, output, mode, in_epoch):

        if isinstance(self.output['gt'], torch.Tensor):
            if global_step % 10 == 0:
                with torch.no_grad():            
                    n_rows, n_cols = 4, 4
                    img_grid = make_grid(self.imgs[0]).cpu().numpy()[0]
                    
                    select_index = np.random.choice(self.imgs[0].shape[-3], n_rows * n_cols, replace = False)
                    imgs_show = self.imgs[0, 0, select_index, ...].cpu().numpy()
                    segs_show = self.output['gt'][0, 1, select_index, ...].cpu().numpy()
                    preds_show = self.output['mask'][0, select_index, ...].cpu().numpy()
                    
                    fig, axes = plt.subplots(n_rows, n_rows, figsize = (12, 12))
                    if not hasattr(axes, 'reshape'):
                        axes = [axes]
                    for i, ax in enumerate(axes.flatten()):
                        ax.axis('off'), ax.set_xticks([]), ax.set_yticks([])
                        ax.imshow(imgs_show[i], cmap='gray')
                        conts_gt = measure.find_contours(segs_show[i].astype(np.uint8))
                        for cont in conts_gt:
                            ax.plot(cont[:, 1], cont[:, 0], linewidth=1, color='#0099ff')
                        conts_pred = measure.find_contours(preds_show[i].astype(np.uint8))
                        for cont in conts_pred:
                            ax.plot(cont[:, 1], cont[:, 0], linewidth=1, color='#ffa500', alpha = 0.5)
                        
                    fig.suptitle('{}_epoch{}.png'.format(mode, global_step))
                    # if not self.training:
                    #     plt.savefig('tmp/check_test_plot/test_plot_{}.png'.format(global_step))

                    writer.add_figure(mode, fig, global_step)
                    if self.cfg.var.obj_operator.is_best:
                        writer.add_figure(f'{mode}_best', fig, global_step)
        
    def filter_vessel(self, seg):

        first_slice = seg[:, 0, :, :].squeeze()

        # apply frangi filter to the first slice
        frangi_slice = frangi(first_slice.detach().cpu().numpy())

        # Assuming `frangi_img` is your frangi filtered image (numpy array)
        threshold_value = np.percentile(frangi_slice, 99.5)  # you can adjust this threshold value
        binary_img = frangi_slice > threshold_value
        filled_img = binary_fill_holes(binary_img)

        # Label connected components
        labeled_img, num_features = label(filled_img)

        # Define function to calculate circularity
        def circularity(area, perimeter):
            if perimeter == 0:
                return 0
            return 4 * np.pi * area / (perimeter ** 2)

        # Extract region properties
        regions = regionprops(labeled_img)

        # Filter regions based on area and circularity
        filtered_regions = [region for region in regions if circularity(region.area, region.perimeter) > 0.95]

        # Sort regions based on area
        sorted_regions = sorted(filtered_regions, key=lambda x: x.area, reverse=True)

        # find the two regions that their center to image center distance are similar while the areas are in top 5
        top_5_regions = sorted_regions[:5]

        # calculate the distance between the center of the region and the center of the image
        center = np.array([first_slice.shape[0] / 2, first_slice.shape[1] / 2])
        center_distance = []
        sum_connected_volume = []
        for region in top_5_regions:
            center_distance.append(np.linalg.norm(np.array(region.centroid) - center))
            seg_copy = seg.detach().cpu().numpy().squeeze().copy()
            # then use flood fill to find the connected components
            seg_copy = flood_fill(seg_copy, (0, int(region.centroid[0]), int(region.centroid[1])), 2)
            seg_copy[seg_copy == 1] = 0
            sum_connected_volume.append(np.sum(seg_copy))

        # selected the index of the two largest connected components
        i, j = np.argsort(sum_connected_volume)[-2:]
        ICA_regions = [top_5_regions[i], top_5_regions[j]]

        # # Check all pairs
        # candidate_pairs = []
        # for i in range(len(top_5_regions)):
        #     for j in range(i+1, len(top_5_regions)):
        #         size_similarity = np.abs(top_5_regions[i].area - top_5_regions[j].area) / np.mean([top_5_regions[i].area, top_5_regions[j].area])
        #         distance_similarity = np.abs(center_distance[i] - center_distance[j]) / np.mean([center_distance[i], center_distance[j]])
        #         distance_diff = np.abs(center_distance[i] - center_distance[j])
                
        #         if size_similarity < 0.1:
        #             candidate_pairs.append([i, j, size_similarity, distance_similarity, distance_diff])

        # ICA_regions = None
        # for i, j, size_similarity, distance_similarity, distance_diff in candidate_pairs:
        #     if distance_diff > 0.2 * first_slice.shape[0]:
        #         continue
        #     else:
        #         ICA_regions = [top_5_regions[i], top_5_regions[j]]
        #         break

        # if not ICA_regions:
        #     i, j = candidate_pairs[0][0], candidate_pairs[0][1]
        #     ICA_regions = [top_5_regions[i], top_5_regions[j]]
        
        # Extract top 2 regions
        # ICA_regions = sorted_regions[:2]

        # Create an empty image to visualize the regions
        output = np.zeros_like(frangi_slice)

        bbox = []
        # Fill the output image with the top 2 regions
        for region in ICA_regions:
            coords = region.coords
            center = region.centroid
            # transfer the tuple to list for center
            center = list(center)
            if center[0] < 0.5 * first_slice.shape[0]:
                bbox_value = [center[0] - region.axis_major_length, center[1] - region.axis_minor_length]
            else:
                bbox_value = [center[0] + region.axis_major_length, center[1] - region.axis_minor_length]
            bbox.append(bbox_value)
            bbox.append([bbox_value[0], first_slice.shape[1]])
            output[coords[:,0], coords[:,1]] = 1

        # plot the first slice of the seg and the frangi slice
        # need to transfer the seg to numpy and also detach
        fig, ax = plt.subplots(1, 3)
        seg_plot = first_slice.detach().cpu().numpy()
        ax[0].imshow(seg_plot, cmap='gray')
        ax[1].imshow(frangi_slice, cmap='gray')
        ax[2].imshow(output, cmap='gray')
        # plot the line that connect the four points in bbox
        # first plot the line that connect the first two points
        ax[0].plot([bbox[0][1], bbox[1][1]], [bbox[0][0], bbox[1][0]], color='r', linewidth=2)
        # then plot the line that connect the first and the third point
        ax[0].plot([bbox[0][1], bbox[2][1]], [bbox[0][0], bbox[2][0]], color='b', linewidth=2)
        # then plot the line that connect the third and the fourth point
        ax[0].plot([bbox[2][1], bbox[3][1]], [bbox[2][0], bbox[3][0]], color='g', linewidth=2)

        # detect all pixels that outside the bbox and then plot them
        # first get the coordinates of all pixels
        x, y = np.where(seg_plot == 1)
        # then get the coordinates of all pixels that outside the bbox
        x_outside = []
        y_outside = []
        for i in range(len(x)):
            if x[i] < min(bbox[0][0], bbox[2][0]) or x[i] > max(bbox[0][0], bbox[2][0]) or y[i] < min(bbox[0][1], bbox[2][1]):
                x_outside.append(x[i])
                y_outside.append(y[i])
        # plot the pixels that outside the bbox
        ax[2].scatter(y_outside, x_outside, color='r', s=1)
        plt.savefig('./tmp/seg_firstslice.png')

        # create a (N,2) array to store the coordinates of the outside points from x_outside and y_outside
        outside_points = np.zeros((len(x_outside), 3))
        outside_points[:, 0] = 0
        outside_points[:, 1] = x_outside
        outside_points[:, 2] = y_outside

        # detect the connected components in seg that connect to the outside_points in the first slice
        # find the connected components looping through all outside_points
        seg_copies = []
        for point in outside_points:
            # use flood fill to find the connected components of the point
            # first create a copy of the seg
            seg_copy = seg.detach().cpu().numpy().squeeze().copy()
            # then use flood fill to find the connected components
            seg_copy = flood_fill(seg_copy, (int(point[0]), int(point[1]), int(point[2])), 2)
            # let the 1 in seg_copy to be 0
            seg_copy[seg_copy == 1] = 0
            seg_copies.append(seg_copy)        

        seg_filter = np.sum(seg_copies, axis=0)
        # return seg if there is no connected components
        if np.sum(seg_filter) == 0:
            print('no output')
            return seg
        seg_filter[seg_filter > 2] = 2

        seg_filter = seg_filter + seg.detach().cpu().numpy().squeeze()
        # 2 for comparison and 0 for final output
        seg_filter[seg_filter == 3] = 0

        # output the seg_copy as nii file
        seg_filter = np.transpose(seg_filter, (1, 2, 0)).astype(np.uint8)
        # seg_nii = nib.Nifti1Image(seg_filter, np.eye(4))
        # nib.save(seg_nii, './tmp/seg_copy.nii.gz')

        # transform seg_filter to tensor to device as same as seg with same shape
        seg_filter = torch.from_numpy(seg_filter.transpose(2,0,1)).unsqueeze(0).to(seg.device)


        return seg_filter