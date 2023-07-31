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
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager

from monai.data import (
    decollate_batch,
)
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, compute_hausdorff_distance
from monai.transforms import AsDiscrete
from monai.losses import DiceLoss, FocalLoss

import torch
import pdb
from torchvision.utils import make_grid
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from einops.einops import rearrange, repeat
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from skimage import measure
from collections import OrderedDict
from sklearn import metrics
from collections import OrderedDict
import torch.distributed as dist
import os
import json
import pandas as pd

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
        self.cldice_loss = soft_dice_cldice()
        self.lumen_loss = LumenLoss(sample_weight = None, pos_weight =1.00, w_ce =0.3, w_dt = 0.3, w_ace=0.5, w_dice=1)

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
        imgs, segs = data['image'], data['label'].long()

        # check if 'weight' is in data
        if 'weight' in data.keys():
            self.sample_weight = data['weight'].view(-1,1).to(self.cfg.var.obj_operator.device)
        else:
            self.sample_weight = None

        # for monai dataset, we usually don't use GPU to accelerate data loading and therefore we will move data to GPU here
        imgs, segs = imgs.to(self.cfg.var.obj_operator.device), segs.to(self.cfg.var.obj_operator.device)
        self.cfg.var.n_samples = b = imgs.shape[0]
        # The gt2onehot function will transform the segmentation into one-hot format
        # The input of the gt2onehot function should be [C, B, ...], where C is the number of classes
        self.output['gt'] = self.gt2onehot(segs.permute(1, 0, 2, 3, 4)).permute(1, 0, 2, 3, 4)
        self.imgs = imgs
        del data, segs, imgs

        if not self.training:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    # sliding window inference is used to get better inference results (from MONAI)
                    self.output['logits'] = sliding_window_inference(self.imgs, roi_size=self.plans['configurations'][self.cfg.exp.nnunet_result.model]['patch_size'], sw_batch_size=4, predictor=self.net)
        else:
            with torch.cuda.amp.autocast():
                self.output['logits'] = self.net(self.imgs)

        # for binary segmentation, the binary function can generate a better segmentation sensitivity
        self.output['mask'] = (self.binary(torch.sigmoid(self.output['logits'][:,1])) > self.threshold).float() # single channel mask
        # transform single channel mask into one-hot format and save as self.output['mask_onehot']
        self.output['mask_onehot'] = self.gt2onehot(self.output['mask'].unsqueeze(0)).permute(1, 0, 2, 3, 4)

        return self.output['logits'][:,1] # actually this return value is not important if you don't use it in get_metrics() function
    
    def loss_focal(self):
        loss = self.focal_loss(self.output['logits'][:, 1], self.output['gt'][:, 1])
        return loss
    
    def loss_cldiceloss(self):
        # the input are in one-hot format but will be transformed into single channel format inside the clDice function
        loss = self.cldice_loss(self.output['gt'], self.output['logits'])
        return loss

    def loss_lumen(self):
        # The input for loss function should be [B, C, ...], where C is 1
        loss = self.lumen_loss(self.imgs, self.output['logits'][:, 1].unsqueeze(1), self.output['gt'][:, 1].unsqueeze(1), self.sample_weight)
        return loss

    def before_epoch(self, mode='train', i_repeat=0):
        self.metrics_epoch = OrderedDict()

        if self.cfg.exp.mode == 'test':
            if self.cfg.exp.test.classification_curve.enable:
                self.gt_roc = []
                self.pred_roc = []

    def after_epoch(self, mode='train'):
        self.metrics_epoch['metric_final'] = 0

        for k, v in self.metrics_epoch.items():
            if self.training:
                self.metrics_epoch[k] = v / len(getattr(self.cfg.var.obj_operator, f'{mode}_set')) * self.cfg.dataset.num_samples
            else:
                self.metrics_epoch[k] = v / len(getattr(self.cfg.var.obj_operator, f'{mode}_set'))
            
        self.metrics_epoch['metric_final'] = self.metrics_epoch['dice']
        
        if self.cfg.exp.mode == 'test':
            if self.cfg.exp.test.classification_curve.enable:
                seg_gt_roc = np.concatenate(self.gt_roc)
                seg_pred_roc = np.concatenate(self.pred_roc)
                fpr, tpr, thres = metrics.roc_curve(y_true=seg_gt_roc.reshape(-1), y_score=seg_pred_roc.reshape(-1))
                df = pd.DataFrame({'fp_rate': fpr, 'tp_rate': tpr, 'threshold': thres})
                df.to_csv(os.path.join(self.cfg.var.obj_operator.path_exp, 'roc_curve_pointwise.csv'), index=False)
                precision, recall, thres = metrics.precision_recall_curve(y_true=seg_gt_roc.reshape(-1),
                                                                          probas_pred=seg_pred_roc.reshape(-1))
                df = pd.DataFrame({
                    'precision': precision,
                    'recall': recall,
                    'threshold': np.concatenate([thres, [99999]])
                })
                df.to_csv(os.path.join(self.cfg.var.obj_operator.path_exp, 'precision_recall_curve_pointwise.csv'),
                          index=False)

                class_gt = np.max(np.max(seg_gt_roc, axis=2), axis=2)[:, 0] # [B]
                class_pred = np.max(np.max(seg_pred_roc, axis=2), axis=2)[:, 0] # [B]
                fpr, tpr, thres = metrics.roc_curve(y_true=class_gt, y_score=class_pred)
                df = pd.DataFrame({'fp_rate': fpr, 'tp_rate': tpr, 'threshold': thres})
                df.to_csv(os.path.join(self.cfg.var.obj_operator.path_exp, 'roc_curve_slicewise.csv'), index=False)
                precision, recall, thres = metrics.precision_recall_curve(y_true=class_gt, probas_pred=class_pred)
                df = pd.DataFrame({
                    'precision': precision,
                    'recall': recall,
                    'threshold': np.concatenate([thres, [99999]])
                })
                df.to_csv(os.path.join(self.cfg.var.obj_operator.path_exp, 'precision_recall_curve_slicewise.csv'),
                          index=False)

                dices = []
                thresholds = np.linspace(np.min(seg_pred_roc), np.max(seg_pred_roc), 80)
                for thre in thresholds:
                    seg_pred_roc_thres = (seg_pred_roc > thre).astype(int)
                    numerator = 2 * np.sum(seg_gt_roc * seg_pred_roc_thres, axis=tuple(range(len(seg_gt_roc.shape))))
                    denominator = np.sum(seg_gt_roc + seg_pred_roc_thres, axis=tuple(range(len(seg_gt_roc.shape))))
                    dice = (numerator + 1e-8) / (denominator + 1e-8)
                    dices.append(dice)
                df = pd.DataFrame({'dice': dices, 'threshold': thresholds})
                df.to_csv(os.path.join(self.cfg.var.obj_operator.path_exp, 'thres_dice_curve.csv'), index=False)

    def get_metrics(self, data, output, mode='train'):
        """
            loss_final: used for backward training
            metric_final: used to select best model (higher is better)
            other metrics: for visualization
        """
        self.metrics_iter = OrderedDict(loss_final=0.)
        for name_loss, w in self.cfg.model.ws_loss.items():
            if w > 0.:
                loss = getattr(self, f'loss_{name_loss}')()
                self.metrics_iter[name_loss] = loss.item()
                self.metrics_iter['loss_final'] += w * loss

        with torch.no_grad():
            seg_gt = self.output['gt'][:,1].type(torch.int64)
            overlapx2, union = self.dice(seg_gt, self.output['mask'], dims_sum=tuple(range(len(seg_gt.shape))), return_before_divide=True)

            self.metrics_iter['dice'] = (overlapx2 + 1e-8) / (union + 1e-8)
            self.metrics_iter['clDice'] = clDice(self.output['mask'].detach().cpu().numpy().squeeze(), seg_gt.detach().cpu().numpy().squeeze())

            if mode in ['val', 'test']:
                self.metrics_iter['ahd'] = ahd_metric(self.output['mask'].detach().cpu().numpy().squeeze(), self.output['gt'][:,1].detach().cpu().numpy().squeeze())
                self.metrics_iter['ahd_monai'] = compute_hausdorff_distance(self.output['mask_onehot'], self.output['gt'])

            if self.cfg.exp.mode == 'test':
                if self.cfg.exp.test.classification_curve.enable:
                    self.gt_roc.append(seg_gt.cpu().numpy())
                    self.pred_roc.append(self.output['logits'].cpu().numpy())

            for k, v in self.metrics_iter.items():
                self.metrics_epoch[k] = self.metrics_epoch.get(k, 0.) + float(v)
                
        return self.metrics_iter

    def vis(self, writer, global_step, data, output, mode, in_epoch):

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
                if not self.training:
                    plt.savefig('tmp/test_plot_{}.png'.format(global_step))

                writer.add_figure(mode, fig, global_step)
                if self.cfg.var.obj_operator.is_best:
                    writer.add_figure(f'{mode}_best', fig, global_step)
        
        