from core.networks.unet import R2U_Net, UNet, NestedUNet
from core.networks.unet3d import Unet3D
# from core.utils.img import get_img_grid, draw_conts_given_mask
from core.losses.DiceLoss import Dice
from core.losses.focal import BinaryFocalLossWithLogits
from core.losses.cldiceLoss import soft_cldice, soft_dice_cldice, soft_dice
from core.utils.clDiceMetric import clDice
from core.utils.bettiMetric import betti_error_metric
from core.utils.ahdMetric import ahd_metric
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager

from monai.data import (
    decollate_batch,
)
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
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
        with open(os.path.join(cfg.exp.nnunet_result.path, 'dataset.json'), 'r') as f:
            dataset_json = json.load(f)
        plans_manager = PlansManager(os.path.join(cfg.exp.nnunet_result.path, 'plans.json'))
        with open(os.path.join(cfg.exp.nnunet_result.path, 'plans.json'), 'r') as f:
                self.plans = json.load(f)

        self.net = get_network_from_plans(plans_manager, 
                                          dataset_json, 
                                          plans_manager.get_configuration(cfg.exp.nnunet_result.model), 
                                          num_input_channels=1)

        id_device = self.cfg.exp.idx_device
        map_location = {'cuda:0': f'cuda:{id_device}'}
        saved_model = torch.load('./save/SMRA_nnUnet.pth')
        pretrained_dict = saved_model['network_weights']
        self.net.load_state_dict(pretrained_dict, strict=True)
        self.net = nnUNet(self.net)
        self.cfg.var.obj_model = self
        self.paths_file_net = []

        self.focal_loss = BinaryFocalLossWithLogits(alpha=cfg.model.w_focal, reduction='mean')
        self.cldice_loss = soft_dice_cldice()

        self.dice = Dice()

        self.gt2onehot = AsDiscrete(to_onehot=2)
        self.pred2onehot = AsDiscrete(to_onehot=2, argmax=True)

        # includes each metric and the total loss for backward training
        self.metrics_iter: dict[str, torch.Tensor | float] = {}
        self.metrics_epoch: dict[str, torch.Tensor | float] = {}
        # [B, M, ...]
        self.imgs: torch.Tensor = None
        # [B, M, ...], ground truth segmentation
        self.segs_gt: torch.Tensor = None
        # [B, 1, ...], predicted segmentation, binary float 
        self.seg_pred: torch.Tensor = None
        self.outputs: dict[str, torch.Tensor] = {}

    def forward(self, data):
        imgs, segs = data['image'], data['label'].long()
        imgs, segs = imgs.to(self.cfg.var.obj_operator.device), segs.to(self.cfg.var.obj_operator.device)
        self.cfg.var.n_samples = b = imgs.shape[0]
        self.segs_gt = self.gt2onehot(segs.permute(1, 0, 2, 3, 4)).permute(1, 0, 2, 3, 4)
        self.imgs = imgs
        del data, segs, imgs

        if not self.training:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    pred = sliding_window_inference(self.imgs, roi_size=self.plans['configurations'][self.cfg.exp.nnunet_result.model]['patch_size'], sw_batch_size=4, predictor=self.net)

        else:
            with torch.cuda.amp.autocast():
                pred = self.net(self.imgs)

        self.outputs['seg'] = torch.sigmoid(pred) # the nnUNet will output results from all decoder layers
        self.seg_pred = (self.outputs['seg'][:,1] > 0.9).float() 
        return self.outputs['seg'][:,1]
    
    def loss_focal(self):
        loss = self.focal_loss(self.outputs['seg'], self.segs_gt)
        return loss
    
    def loss_cldiceloss(self):
        loss = self.cldice_loss(self.segs_gt, self.outputs['seg'])
        return loss

    def before_epoch(self, mode='train', i_repeat=0):
        self.metrics_epoch = OrderedDict()

    def after_epoch(self, mode='train'):
        self.metrics_epoch['metric_final'] = 0

        for k, v in self.metrics_epoch.items():
            self.metrics_epoch[k] = v / len(getattr(self.cfg.var.obj_operator, f'{mode}_set'))
            
        self.metrics_epoch['metric_final'] = self.metrics_epoch['dice']
        
        if self.cfg.exp.mode == 'test':
            if self.cfg.exp.test.classification_curve.enable:
                seg_gt_roc = np.concatenate(self.segs_gt_roc)
                seg_pred_roc = np.concatenate(self.segs_pred_roc)
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
                # print('loss: ', self.metrics_iter['loss_final'])

        with torch.no_grad():
            seg_gt = self.segs_gt[:,1].type(torch.int64)
            overlapx2, union = self.dice(seg_gt, self.seg_pred, dims_sum=tuple(range(len(seg_gt.shape))), return_before_divide=True)

            self.metrics_iter['dice'] = (overlapx2 + 1e-8) / (union + 1e-8)
            self.metrics_iter['clDice'] = clDice(self.seg_pred.detach().cpu().numpy().squeeze(), seg_gt.detach().cpu().numpy().squeeze())

            if mode in ['val', 'test']:
                self.metrics_iter['ahd'] = ahd_metric(self.seg_pred.detach().cpu().numpy().squeeze(), self.segs_gt[:,1].detach().cpu().numpy().squeeze())

            if self.cfg.exp.mode == 'test':
                if self.cfg.exp.test.classification_curve.enable:
                    self.segs_gt_roc.append(seg_gt.cpu().numpy())
                    self.segs_pred_roc.append(self.outputs['seg'].cpu().numpy())

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
                segs_show = self.segs_gt[0, 1, select_index, ...].cpu().numpy()
                preds_show = self.seg_pred[0, select_index, ...].cpu().numpy()
                
                fig, axes = plt.subplots(n_rows, n_rows, figsize = (12, 12))
                if not hasattr(axes, 'reshape'):
                    axes = [axes]
                for i, ax in enumerate(axes.flatten()):
                    ax.axis('off'), ax.set_xticks([]), ax.set_yticks([])
                    ax.imshow(imgs_show[..., i], cmap='gray')
                    conts_gt = measure.find_contours(segs_show[..., i].astype(np.uint8))
                    for cont in conts_gt:
                        ax.plot(cont[:, 1], cont[:, 0], linewidth=1, color='#0099ff')
                    conts_pred = measure.find_contours(preds_show[..., i].astype(np.uint8))
                    for cont in conts_pred:
                        ax.plot(cont[:, 1], cont[:, 0], linewidth=1, color='#ffa500', alpha = 0.5)
                    
                fig.suptitle('{}_epoch{}.png'.format(mode, global_step))
                # plt.savefig('tmp/test_plot.png')

                writer.add_figure(mode, fig, global_step)
                if self.cfg.var.obj_operator.is_best:
                    writer.add_figure(f'{mode}_best', fig, global_step)
        
        