from core.networks.unet import R2U_Net, UNet, NestedUNet
from core.networks.unet3d import Unet3D
from core.losses.TopoInter_loss import TopoInter_Loss
from core.utils.clDiceMetric import clDice
from core.utils.bettiMetric import betti_error_metric
from core.utils.ahdMetric import ahd_metric
from core.utils.ahdgpuMetric import ahdgpu_metric
from core.utils.CoWMetric import betti_number, betti_number_error_all_classes

from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager

from monai.data import (
    decollate_batch,
)
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, compute_hausdorff_distance
from monai.transforms import AsDiscrete
from monai.losses import DiceLoss, GeneralizedDiceLoss
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
import matplotlib.colors as mcolors
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

class nntopoModel(nn.Module):

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

        self.exclusion_list = [[1,4], [1,5], [1,6], [1,7], [1,8], [1,9], [1,10], [1,11], [1,12], [1,13],
                               [2,3], [2,4], [2,5], [2,6], [2,7], [2,9], [2,10], [2,11], [2,12], [2,13],
                               [3,4], [3,5], [3,6], [3,7], [3,8], [3,10], [3,11], [3,12], [3,13],
                               [4,6], [4,7], [4,9], [4,10], [4,12], [4,13],
                               [5,6], [5,7], [5,8],[5,9], [5,10], [5,11],[5,12], [5,13],
                               [6,8], [6,10], [6,11], [6,13],
                               [7,8],[7,9], [7,10], [7,11],[7,12], [7,13],
                               [8,9], [8,10], [8,11],[8,12], [8,13],
                               [9,10], [9,11],[9,12], [9,13],
                               [11,12]]

        # Here we import the different metrics
        self.dice_loss = GeneralizedDiceLoss(to_onehot_y = True, softmax=True, include_background=False)
        self.topo_loss = TopoInter_Loss(dim=3, connectivity=26, inclusion=[], exclusion=self.exclusion_list, min_thick=1)
        self.dice = DiceMetric(include_background=True, reduction="mean", get_not_nans=True)

        self.gt2onehot = AsDiscrete(to_onehot=14)
        self.pred2onehot = AsDiscrete(to_onehot=14, argmax=True)

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

        self.cmap = mcolors.ListedColormap(['#000000', '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF', '#FF00FF', 
                               '#800000', '#008000', '#000080', '#808000', '#800080', '#008080', '#C0C0C0', '#808080'])

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
            self.output['gt_singlechannel'] = segs
        else:
            imgs = data['image']
            imgs = imgs.to(self.cfg.var.obj_operator.device)
            segs = None
        
        # self.output['image_meta_dict'] = data['image_meta_dict']

        self.imgs = imgs

        self.cfg.var.n_samples = b = imgs.shape[0]
        
        if not self.training:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    # sliding window inference is used to get better inference results (from MONAI)
                    self.output['logits'] = sliding_window_inference(self.imgs, roi_size=self.plans['configurations'][self.cfg.exp.nnunet_result.model]['patch_size'], 
                                                                        sw_batch_size=4, predictor=self.net, overlap=0.1, mode='gaussian', padding_mode='reflect')
        else:
            with torch.cuda.amp.autocast():
                self.output['logits'] = self.net(self.imgs)

        # for multi-class segmentation, we will use softmax and then convert it to onehot to get the predicted segmentation
        self.output['mask'] = self.pred2onehot(F.softmax(self.output['logits'], dim=1).permute(1, 0, 2, 3, 4)).permute(1, 0, 2, 3, 4)
        self.output['mask_singlechannel'] = torch.argmax(self.output['mask'], dim=1, keepdim=True)
        
        del data, segs, imgs
    
    def loss_dice(self):
        loss = self.dice_loss(self.output['logits'], self.output['gt_singlechannel'])
        return loss
    
    def loss_topo(self):
        loss = self.topo_loss(self.output['logits'], self.output['gt_singlechannel'])
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
                self.metrics_epoch['metric_final'] = self.metrics_epoch['dice']

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
                
                self.metrics_iter['dice'] = self.dice(self.output['mask'].detach().cpu(), self.output['gt'].detach().cpu())
                # if metrics_iter['dice'] is not one value, manually calculate the mean and exclude the nan values
                # the shape of the self.metrics_iter['dice'] is [batch_size, n_classes]
                if self.metrics_iter['dice'].shape[1] > 1:
                    count = torch.sum(~torch.isnan(self.metrics_iter['dice']), dim=1)
                    # first calculate the mean dice of each case in the batch which will exclude all nan values and then calculate the mean of the batch
                    mean_dice_per_case = torch.nansum(self.metrics_iter['dice'], dim=1) / count
                    self.metrics_iter['dice'] = torch.mean(mean_dice_per_case)
                self.metrics_iter['cldice'] = clDice(np.where(self.output['mask_singlechannel'].detach().cpu().numpy().squeeze()>0, True, False).astype(np.uint8), np.where(self.output['gt_singlechannel'].detach().cpu().numpy().squeeze()>0, True, False).astype(np.uint8))
                # self.metrics_iter['betti_error'] = betti_number_error_all_classes()

                # temporary save the predicted mask and ground truth
                # seg = self.output['mask_singlechannel'].detach().cpu().numpy().squeeze().astype(np.uint8)
                # baseid = self.output['image_meta_dict']['filename_or_obj'][0].split('/')[-1].split('.')[0]
                # seg = np.transpose(seg, (1, 2, 0))
                # # flip the last axis of seg only for CoW data
                # seg = np.flip(seg, axis=2)
                # seg = nib.Nifti1Image(seg, np.eye(4))
                # # save the seg file with the same name as in the image_meta_dict
                # nib.save(seg, os.path.join(self.cfg.var.obj_operator.path_exp, baseid + '.nii.gz'))

                if mode in ['val', 'test']:
                    pass
                        
            for k, v in self.metrics_iter.items():
                self.metrics_epoch[k] = self.metrics_epoch.get(k, 0.) + float(v) * self.imgs.shape[0]

        if self.cfg.exp.mode == 'test':

            check = False
            if check:

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
            # with open(os.path.join(self.cfg.var.obj_operator.path_exp, 'metrics.txt'), 'a') as f:
            #     f.write(self.output['image_meta_dict']['filename_or_obj'][0].split('/')[-1].split('.')[0] + '\n')
            #     f.write(str(self.metrics_iter) + '\n')                
            
            # print the case name
            print(self.output['image_meta_dict']['filename_or_obj'][0].split('/')[-1].split('.')[0])
            # print the items in the self.metrics_iter
            for k, v in self.metrics_iter.items():
                print(k, v)

        return self.metrics_iter

    def vis(self, writer, global_step, data, output, mode, in_epoch):

        if isinstance(self.output['gt'], torch.Tensor): # not plot for test
            if global_step % 10 == 0:
                with torch.no_grad():            
                    n_rows, n_cols = 4, 4
                    img_grid = make_grid(self.imgs[0]).cpu().numpy()[0]

                    select_index = np.random.choice(self.imgs[0].shape[-3], n_rows * n_cols, replace = False)
                    imgs_show = self.imgs[0, 0, select_index, ...].cpu().numpy()
                    segs_show = self.output['gt_singlechannel'][0, 0, select_index, ...].cpu().numpy()
                    preds_show = self.output['mask_singlechannel'][0,0, select_index, ...].cpu().numpy()
                    
                    fig, axes = plt.subplots(n_rows, n_rows, figsize = (12, 12))
                    axes = axes.flatten()
                    # Split axes into two parts
                    axes_gt = axes[:len(axes)//2]
                    axes_pred = axes[len(axes)//2:]

                    # First two rows for ground truth
                    for i, ax in enumerate(axes_gt):
                        ax.axis('off'), ax.set_xticks([]), ax.set_yticks([])
                        ax.imshow(imgs_show[i], cmap='gray')
                        conts_gt = measure.find_contours(segs_show[i].astype(np.uint8), level=0.5)
                        for cont in conts_gt:
                            ax.plot(cont[:, 1], cont[:, 0], linewidth=1, color=self.cmap.colors[int(segs_show[i][int(cont[0][0]), int(cont[0][1])])])

                    # Last two rows for prediction
                    for i, ax in enumerate(axes_pred):
                        ax.axis('off'), ax.set_xticks([]), ax.set_yticks([])
                        ax.imshow(imgs_show[i], cmap='gray')
                        conts_pred = measure.find_contours(preds_show[i].astype(np.uint8), level=0.5)
                        for cont in conts_pred:
                            ax.plot(cont[:, 1], cont[:, 0], linewidth=1, color=self.cmap.colors[int(preds_show[i][int(cont[0][0]), int(cont[0][1])])], alpha = 0.5)
                        
                    fig.suptitle('{}_epoch{}.png'.format(mode, global_step))
                    # if not self.training:
                    #     plt.savefig('tmp/check_test_plot/test_plot_CoWSemantic.png'.format(global_step))
                    # pdb.set_trace()

                    writer.add_figure(mode, fig, global_step)
                    if self.cfg.var.obj_operator.is_best:
                        writer.add_figure(f'{mode}_best', fig, global_step)
        