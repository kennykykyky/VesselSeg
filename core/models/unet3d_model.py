from core.networks.unet import R2U_Net, UNet, NestedUNet
from core.networks.unet3d import Unet3D
# from core.utils.img import get_img_grid, draw_conts_given_mask
from core.losses.DiceLoss import Dice
from core.losses.focal import BinaryFocalLossWithLogits
from core.losses.cldiceLoss import soft_cldice, soft_dice_cldice, soft_dice
from core.utils.clDiceMetric import clDice
from core.utils.bettiMetric import betti

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

mpl.use('agg')

class UNet3DModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.net = Unet3D(cfg)
        self.cfg.var.obj_model = self
        self.paths_file_net = ['core/networks/unet3d.py']

        self.focal_loss = BinaryFocalLossWithLogits(alpha=cfg.model.w_focal, reduction='mean')
        # self.cldice_loss = soft_cldice()
        # self.cldice_loss = soft_dice_cldice()
        self.dice = Dice()
        # includes each metric and the total loss for backward training
        self.metrics_iter: dict[str, torch.Tensor | float] = {}
        self.metrics_epoch: dict[str, torch.Tensor | float] = {}
        # [B, M, ...]
        self.imgs: torch.Tensor = None
        # [B, M, ...], ground truth segmentation
        self.segs_gt: torch.Tensor = None
        # [B, 1, ...], predicted segmentation, binary float 
        self.seg_pred: torch.Tensor = None

    def forward(self, data):
        imgs, segs = data['imgs'], data['segs']
        self.cfg.var.n_samples = b = imgs.shape[0]
        self.imgs = imgs
        self.segs_gt = segs
        output = self.net(self.imgs)
        self.seg_pred = (self.net.outputs['seg'] > 0.9).float() 
        return output

    def loss_focal(self):
        loss = self.focal_loss(self.net.outputs['seg'], self.segs_gt[:, None])
        # loss = self.cldice_loss(self.net.outputs['seg'], self.segs_gt[:, None])
        return loss

    def before_epoch(self, mode='train', i_repeat=0):
        self.metrics_epoch = OrderedDict()

    def after_epoch(self, mode='train'):
        self.metrics_epoch['metric_final'] = 0
        for k, v in self.metrics_epoch.items():
            if k[:4] != 'seg_':
                self.metrics_epoch[k] = v / len(getattr(self.cfg.var.obj_operator, f'{mode}_set'))
        overlapx2 = self.metrics_epoch['seg_overlapx2']
        union = self.metrics_epoch['seg_union']
        self.metrics_epoch['dice'] = (overlapx2 + 1e-8) / (union + 1e-8)

        if mode in ['val', 'test']:
            tp = self.metrics_epoch['seg_tp_pointwise']
            tn = self.metrics_epoch['seg_tn_pointwise']
            fp = self.metrics_epoch['seg_fp_pointwise']
            fn = self.metrics_epoch['seg_fn_pointwise']
            self.metrics_epoch['sensitivity_pointwise'] = se = tp / (tp + fn + 1e-8)
            self.metrics_epoch['precision_pointwise'] = pc = tp / (tp + fp + 1e-8)
            self.metrics_epoch['f1_score_pointwise'] = 2 * pc * se / (pc + se + 1e-8)
            self.metrics_epoch['specificity_pointwise'] = tn / (tn + fp + 1e-8)

            tp = self.metrics_epoch['seg_tp_slicewise']
            tn = self.metrics_epoch['seg_tn_slicewise']
            fp = self.metrics_epoch['seg_fp_slicewise']
            fn = self.metrics_epoch['seg_fn_slicewise']
            self.metrics_epoch['sensitivity_slicewise'] = se = tp / (tp + fn + 1e-8)
            self.metrics_epoch['precision_slicewise'] = pc = tp / (tp + fp + 1e-8)
            self.metrics_epoch['f1_score_slicewise'] = 2 * pc * se / (pc + se + 1e-8)
            self.metrics_epoch['specificity_slicewise'] = tn / (tn + fp + 1e-8)
            
            self.metrics_epoch['clDice'] = clDice(np.squeeze(self.seg_pred.cpu().numpy()), np.squeeze(self.segs_gt.cpu().numpy()))
            # self.metrics_epoch['betti'] = betti(np.squeeze(self.seg_pred.cpu().numpy()), np.squeeze(self.segs_gt.cpu().numpy()))

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

        with torch.no_grad():
            seg_gt = self.segs_gt.type(torch.int64)[:, None] # [B, 1, ...]
            overlapx2, union = self.dice(seg_gt, self.seg_pred, dims_sum=tuple(range(len(seg_gt.shape))),
                                         return_before_divide=True)

            self.metrics_iter['seg_overlapx2'] = overlapx2
            self.metrics_iter['seg_union'] = union
            self.metrics_iter['seg_n_pixels_gt'] = torch.sum(seg_gt)

            if mode in ['test', 'val']:
                self.metrics_iter['seg_tp_pointwise'] = tp = overlapx2 / 2
                self.metrics_iter['seg_fp_pointwise'] = torch.sum(self.seg_pred) - tp
                self.metrics_iter['seg_fn_pointwise'] = torch.sum(seg_gt) - tp
                self.metrics_iter['seg_tn_pointwise'] = torch.prod(torch.tensor(seg_gt.shape)) - union + tp

                class_gt = torch.any(torch.any(seg_gt, dim=2), dim=2)[:, 0] # [B]
                class_pred = torch.any(torch.any(self.seg_pred, dim=2), dim=2)[:, 0] # [B]
                self.metrics_iter['seg_tp_slicewise'] = tp = torch.sum(class_gt * class_pred)
                self.metrics_iter['seg_fp_slicewise'] = torch.sum(class_pred) - tp
                self.metrics_iter['seg_fn_slicewise'] = torch.sum(class_gt) - tp
                self.metrics_iter['seg_tn_slicewise'] = len(class_gt) - torch.sum(class_gt + class_pred) + tp

            if self.cfg.exp.mode == 'test':
                if self.cfg.exp.test.classification_curve.enable:
                    self.segs_gt_roc.append(seg_gt.cpu().numpy())
                    self.segs_pred_roc.append(self.net.decoder.outputs['seg'].cpu().numpy())

            for k, v in self.metrics_iter.items():
                if k[:4] == 'seg_':
                    self.metrics_epoch[k] = self.metrics_epoch.get(k, 0.) + float(v)
                else:
                    self.metrics_epoch[k] = self.metrics_epoch.get(k, 0.) + float(v) * self.cfg.var.n_samples
        return self.metrics_iter

    def vis(self, writer, global_step, data, output, mode, in_epoch):

        if global_step % 10 == 0:
            with torch.no_grad():            
                n_rows, n_cols = 4, 4
                img_grid = make_grid(self.imgs[0]).cpu().numpy()[0]
                
                select_index = np.random.choice(self.imgs[0].shape[-1], n_rows * n_cols, replace = False)
                imgs_show = self.imgs[0, 0, ..., select_index].cpu().numpy()
                segs_show = self.segs_gt[0, ..., select_index].cpu().numpy()
                preds_show = self.seg_pred[0, 0, ..., select_index].cpu().numpy()
                
                fig, axes = plt.subplots(n_rows, n_rows, figsize = (12, 12))
                if not hasattr(axes, 'reshape'):
                    axes = [axes]
                for i, ax in enumerate(axes.flatten()):
                    ax.axis('off'), ax.set_xticks([]), ax.set_yticks([])
                    ax.imshow(imgs_show[..., i], cmap='gray')
                    conts_gt = measure.find_contours(segs_show[..., i])
                    for cont in conts_gt:
                        ax.plot(cont[:, 1], cont[:, 0], linewidth=1, color='#0099ff')
                    conts_pred = measure.find_contours(preds_show[..., i])
                    for cont in conts_pred:
                        ax.plot(cont[:, 1], cont[:, 0], linewidth=1, color='#ffa500', alpha = 0.5)
                    
                fig.suptitle('{}_epoch{}.png'.format(mode, global_step))
                # plt.savefig('tmp/test_plot.png')

                writer.add_figure(mode, fig, global_step)
                if self.cfg.var.obj_operator.is_best:
                    writer.add_figure(f'{mode}_best', fig, global_step)
                    
                # if global_step == 200:
                #         pdb.set_trace()
        
        