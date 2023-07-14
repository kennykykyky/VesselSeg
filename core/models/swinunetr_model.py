from core.networks.unet import R2U_Net, UNet, NestedUNet
from core.networks.unet3d import Unet3D
# from core.utils.img import get_img_grid, draw_conts_given_mask
from core.losses.DiceLoss import Dice
from core.losses.focal import BinaryFocalLossWithLogits

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
import os
import pandas as pd
from monai.networks.nets import SwinUNETR

from monai.data import (
    decollate_batch,
)
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.losses import DiceLoss, FocalLoss

mpl.use('agg')

class swinunetrModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.net = SwinUNETR(img_size=cfg.net.patch_size, in_channels=cfg.net.in_channels, out_channels=cfg.net.out_channels, feature_size=cfg.net.feature_size, use_checkpoint=cfg.net.use_checkpoint)
        self.cfg.var.obj_model = self
        self.paths_file_net = []

        # self.focal_loss = BinaryFocalLossWithLogits(alpha=cfg.model.w_focal, reduction='mean')
        self.focal_loss = FocalLoss(alpha=cfg.model.w_focal, reduction='mean')
        self.dice_loss = DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
        
        self.metrics_iter: dict[str, torch.Tensor | float] = {}
        self.metrics_epoch: dict[str, torch.Tensor | float] = {}
        self.metrics_dice = DiceMetric(include_background=False, reduction="mean", get_not_nans = False)
        
        self.output = {}
        
        self.gt2onehot = AsDiscrete(to_onehot=2)
        self.pred2onehot = AsDiscrete(to_onehot=2, argmax=True)
        
        if not self.cfg.exp.train.path_model_trained:
            weight = torch.load("save/model_swinvit.pt")
            self.net.load_from(weights=weight)
                
    def forward(self, data):
        img, gt = data['image'], data['label']
        
        self.imgs = img
        self.segs_gt = gt
        
        if not self.training:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    pred = sliding_window_inference(img, roi_size=self.cfg.net.patch_size, sw_batch_size=4, predictor=self.net)
                    
                gts = decollate_batch(gt)
                preds = decollate_batch(pred)
                self.seg_pred = pred
                onehots_gt = [self.gt2onehot(gt) for gt in gts]
                onehots_pred = [self.pred2onehot(pred) for pred in preds]
                self.output = {'onehots_gt': onehots_gt, 'onehots_pred': onehots_pred}
                return self.output
        else:
            with torch.cuda.amp.autocast():
                pred = self.net(img)
                self.seg_pred = pred
            self.output = {'logits_pred': pred, 'logits_gt': gt}
            return self.output

    def loss_focal(self):
        loss = self.focal_loss(self.output['logits_pred'], self.output['logits_gt'])
        return loss
    
    def loss_diceloss(self):
        loss = self.dice_loss(self.output['logits_pred'], self.output['logits_gt'])
        return loss

    def before_epoch(self, mode='train', i_repeat=0):
        self.mode = mode
        self.metrics_epoch = OrderedDict()
        if not self.training:
            self.metrics_dice.reset()

    def after_epoch(self, mode='train'):
        for k, v in self.metrics_epoch.items():
            self.metrics_epoch[k] = v / len(getattr(self.cfg.var.obj_operator, f'{mode}_set'))
        if not self.training:
            self.metrics_epoch['dice_mean'] = self.metrics_dice.aggregate().item()
            self.metrics_epoch['metric_final'] = self.metrics_epoch['dice_mean']
            self.metrics_dice.reset()
        # del self.output

    def get_metrics(self, data, output, mode='train'):
        """
            loss_final: used for backward training
            metric_final: used to select best model (higher is better)
            other metrics: for visualization
        """
        self.metrics_iter = OrderedDict()
        if self.training:
            self.metrics_iter['loss_final'] = 0.
            for name_loss, w in self.cfg.model.ws_loss.items():
                if w > 0.:
                    loss = getattr(self, f'loss_{name_loss}')()
                    self.metrics_iter[name_loss] = loss.item()
                    self.metrics_iter['loss_final'] += w * loss

        with torch.no_grad():
            if not self.training:
                self.metrics_dice(y_pred=output['onehots_pred'], y=output['onehots_gt'])
                
        return self.metrics_iter

    def vis(self, writer, global_step, data, output, mode, in_epoch):
        pass

        # if global_step % 10 == 0:
        #     with torch.no_grad():            
        #         n_rows, n_cols = 4, 4
        #         img_grid = make_grid(self.imgs[0]).cpu().numpy()[0]
                
        #         select_index = np.random.choice(self.imgs[0].shape[-1], n_rows * n_cols, replace = False)
        #         imgs_show = self.imgs[0, 0, ..., select_index].cpu().numpy()
        #         segs_show = self.segs_gt[0, ..., select_index].cpu().numpy()
        #         preds_show = self.seg_pred[0, 0, ..., select_index].cpu().numpy()
                
        #         fig, axes = plt.subplots(n_rows, n_rows, figsize = (12, 12))
        #         if not hasattr(axes, 'reshape'):
        #             axes = [axes]
        #         for i, ax in enumerate(axes.flatten()):
        #             ax.axis('off'), ax.set_xticks([]), ax.set_yticks([])
        #             ax.imshow(imgs_show[..., i], cmap='gray')
        #             conts_gt = measure.find_contours(segs_show[..., i])
        #             for cont in conts_gt:
        #                 ax.plot(cont[:, 1], cont[:, 0], linewidth=1, color='#0099ff')
        #             conts_pred = measure.find_contours(preds_show[..., i])
        #             for cont in conts_pred:
        #                 ax.plot(cont[:, 1], cont[:, 0], linewidth=1, color='#ffa500', alpha = 0.5)
                    
        #         fig.suptitle('{}_epoch{}.png'.format(mode, global_step))
        #         # plt.savefig('tmp/test_plot.png')

        #         writer.add_figure(mode, fig, global_step)
        #         if self.cfg.var.obj_operator.is_best:
        #             writer.add_figure(f'{mode}_best', fig, global_step)
                    
        #         # if global_step == 200:
        #         #         pdb.set_trace()
