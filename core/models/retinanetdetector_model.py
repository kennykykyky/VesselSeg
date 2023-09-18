from core.networks.unet import R2U_Net, UNet, NestedUNet
from core.networks.unet3d import Unet3D
# from core.utils.img import get_img_grid, draw_conts_given_bbox
from core.losses.boxLoss import BoxLoss
from core.utils.clDiceMetric import clDice
from core.utils.bettiMetric import betti_error_metric
from core.utils.ahdMetric import ahd_metric
from core.utils.ahdgpuMetric import ahdgpu_metric
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager

from monai.data import box_utils
from monai.inferers import sliding_window_inference
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
from monai.networks.nets import resnet
from monai.apps.detection.metrics.coco import COCOMetric
from monai.apps.detection.metrics.matching import matching_batch
from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
from monai.apps.detection.networks.retinanet_network import (
    RetinaNet,
    resnet_fpn_feature_extractor,
)

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
import matplotlib.patches as patches
from skimage.measure import regionprops
from skimage.segmentation import flood_fill

mpl.use('agg')

class retinanetdetectorModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.cfg.var.obj_model = self
        self.paths_file_net = []

        self.spatial_dims = 3

        # Here we import the different losses
        self.box_loss = BoxLoss(self.spatial_dims)

        # includes each metric and the total loss for backward training
        self.metrics_iter: dict[str, torch.Tensor | float] = {}
        self.metrics_epoch: dict[str, torch.Tensor | float] = {}
        # [B, M, ...]
        self.imgs: torch.Tensor = None
        self.output: dict[str, torch.Tensor] = {}
        # [B, M, ...], ground truth segmentation
        self.output['gt']: torch.Tensor = None
        # [B, 1, ...], predicted segmentation, binary float 
        self.output['box_loss']: torch.Tensor = None

        self.val_output_all: dict[str, torch.Tensor] = {}
        self.val_output_all['preds'] = []
        self.val_output_all['gt'] = []

        # 1) build anchor generator
        # returned_layers: when target boxes are small, set it smaller
        # base_anchor_shapes: anchor shape for the most high-resolution output,
        #   when target boxes are small, set it smaller

        anchor_generator = AnchorGeneratorWithAnchorShape(
            feature_map_scales=[2**l for l in range(len(cfg.net.returned_layers) + 1)],
            base_anchor_shapes=cfg.net.base_anchor_shapes,
        )
        # anchor_generator = AnchorGeneratorWithAnchorShape(feature_map_scales=cfg.net.returned_layers, base_anchor_shapes=cfg.net.base_anchor_shapes,)

        # 2) build network
        conv1_t_size = [max(7, 2 * s + 1) for s in cfg.net.conv1_t_stride]
        backbone = resnet.ResNet(
            block=resnet.ResNetBottleneck,
            layers=[3, 4, 6, 3],
            block_inplanes=resnet.get_inplanes(),
            n_input_channels=cfg.net.n_input_channels,
            conv1_t_stride=cfg.net.conv1_t_stride,
            conv1_t_size=conv1_t_size,
        )
        feature_extractor = resnet_fpn_feature_extractor(
            backbone=backbone,
            spatial_dims=cfg.net.spatial_dims,
            pretrained_backbone=False,
            trainable_backbone_layers=None,
            returned_layers=cfg.net.returned_layers,
        )
        num_anchors = anchor_generator.num_anchors_per_location()[0]
        size_divisible = [s * 2 * 2 ** max(cfg.net.returned_layers) for s in feature_extractor.body.conv1.stride]

        self.net = torch.jit.script(
            RetinaNet(
                spatial_dims=cfg.net.spatial_dims,
                num_classes=1,
                num_anchors=num_anchors,
                feature_extractor=feature_extractor,
                size_divisible=size_divisible,
            )
        )

        # 3) build detector
        self.detector = RetinaNetDetector(network=self.net, anchor_generator=anchor_generator, debug=False).to(self.cfg.var.obj_operator.device)

        # set training components
        self.detector.set_atss_matcher(num_candidates=1, center_in_gt=False)
        self.detector.set_hard_negative_sampler(
            batch_size_per_image=8, # number of proposals
            positive_fraction=cfg.net.balanced_sampler_pos_fraction,
        )
        self.detector.set_target_keys(box_key="box", label_key="label")

        # set validation components
        self.detector.set_box_selector_parameters(
            score_thresh=cfg.net.score_thresh,
            topk_candidates_per_level=1,
            nms_thresh=cfg.net.nms_thresh,
            detections_per_img=1,
        )
        self.detector.set_sliding_window_inferer(
            roi_size=cfg.net.val_patch_size,
            overlap=0.25,
            sw_batch_size=4,
            mode="constant",
            device=self.cfg.var.obj_operator.device,
        )

        self.coco_metric = COCOMetric(classes=["CoW"], iou_list=[0.5], max_detection=[10])

    def forward(self, data):

        # if data is a list, extract all 'image' features from each item in the list
        if isinstance(data, list):
            if not isinstance(data[0], list):
                data = [[item] for item in data] # for test and val with batch size 1
            imgs = [item['image'].to(self.cfg.var.obj_operator.device) for data1 in data for item in data1]
            segs = [item['seg'].to(self.cfg.var.obj_operator.device) for data1 in data for item in data1]
            labels = [item['label'].to(self.cfg.var.obj_operator.device).long() for data1 in data for item in data1]
            bboxs = [item['box'].to(self.cfg.var.obj_operator.device) for data1 in data for item in data1]
        
        # concat the imgs and segs along axis 1 to form self.input
        self.input = [torch.cat((img, seg), dim=0) for img, seg in zip(imgs, segs)]

        # create self.target as a dictionary that contains the labels and the bboxs
        self.target = [{'label': label, 'box': bbox} for label, bbox in zip(labels, bboxs)]
        self.output['gt'] = [item['box'] for item in self.target]

        self.cfg.var.n_samples = b = len(imgs)
        
        if not self.training:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    use_inferer = not all(
                        [val_data_i[0, ...].numel() < np.prod(self.cfg.net.val_patch_size) for val_data_i in self.input]
                    )
                    self.output['preds'] = self.detector(self.input, use_inferer=use_inferer)
                    self.val_output_all['preds'] += self.output['preds']
                    self.val_output_all['gt'] += self.target
        else:
            with torch.cuda.amp.autocast():
                self.output['box_loss'] = self.detector(self.input, self.target)['box_regression']

        del data, bboxs, imgs, labels

        return self.output # actually this return value is not important if you don't use it in get_metrics() function
    
    def before_epoch(self, mode='train', i_repeat=0):
        self.metrics_epoch = OrderedDict()

        if self.cfg.exp.mode == 'test':
            if self.cfg.exp.test.classification_curve.enable:
                self.gt_roc = []
                self.pred_roc = []

    def after_epoch(self, mode='train'):

        for k, v in self.metrics_epoch.items():
            if self.training:
                self.metrics_epoch[k] = v / len(getattr(self.cfg.var.obj_operator, f'{mode}_set')) / self.cfg.dataset.num_samples
            else:
                self.metrics_epoch[k] = v / len(getattr(self.cfg.var.obj_operator, f'{mode}_set'))

        if not self.training:
            with torch.no_grad():
                results_metric = matching_batch(
                    iou_fn=box_utils.box_iou,
                    iou_thresholds=self.coco_metric.iou_thresholds,
                    pred_boxes=[
                        val_data_i[self.detector.target_box_key].cpu().detach().numpy() for val_data_i in self.val_output_all['preds']
                    ],
                    pred_classes=[
                        val_data_i[self.detector.target_label_key].cpu().detach().numpy() for val_data_i in self.val_output_all['preds']
                    ],
                    pred_scores=[
                        val_data_i[self.detector.pred_score_key].cpu().detach().numpy() for val_data_i in self.val_output_all['preds']
                    ],
                    gt_boxes=[val_data_i[self.detector.target_box_key].cpu().detach().numpy() for val_data_i in self.val_output_all['gt']],
                    gt_classes=[
                        val_data_i[self.detector.target_label_key].cpu().detach().numpy() for val_data_i in self.val_output_all['gt']
                    ],
                )
                val_epoch_metric_dict = self.coco_metric(results_metric)[0]
                print(val_epoch_metric_dict)
            
            # write each item in self.metrics_epoch
            for k, v in val_epoch_metric_dict.items():
                self.metrics_epoch[k] = v

            self.metrics_epoch['metric_final'] = self.metrics_epoch['AP_IoU_0.50_MaxDet_10']

    def loss_box(self):
        return self.output['box_loss']

    def get_metrics(self, data, output, mode='train'):
        """
            loss_final: used for backward training
            metric_final: used to select best model (higher is better)
            other metrics: for visualization
        """
        self.metrics_iter = OrderedDict(loss_final=0.)

        # test plot
        # bbox = output['gt'][0][0].cpu().numpy().astype(int)
        # image = self.input[0][0].cpu().numpy()
        # slices = np.linspace(bbox[2], bbox[-1], 5, dtype=int)
        # fig, axs = plt.subplots(5, 2, figsize=(10, 10))
        # for i, slice in enumerate(slices):
        #     axs[i, 0].imshow(image[:, :, slice], cmap='gray')
        #     axs[i, 0].add_patch(patches.Rectangle((bbox[1], bbox[0]), bbox[4]-bbox[1], bbox[3]-bbox[0], linewidth=1, edgecolor='r', facecolor='none'))
        #     axs[i, 1].imshow(image[int(bbox[0]):int(bbox[3]), int(bbox[1]):int(bbox[4]), slice], cmap='gray')
        # plt.savefig('./tmp/Cow/check_in_NN.png')

        if self.training:
            for name_loss, w in self.cfg.model.ws_loss.items():
                if w > 0.:
                    loss = getattr(self, f'loss_{name_loss}')()
                    self.metrics_iter[name_loss] = loss.item()
                    self.metrics_iter['loss_final'] += w * loss

        for k, v in self.metrics_iter.items():
            self.metrics_epoch[k] = self.metrics_epoch.get(k, 0.) + float(v) * len(self.input)

        if self.cfg.exp.mode == 'test':

            check = True
            if check:
                pass

            if self.cfg.exp.test.save_seg.enable:
                # save image as nii file
                pass
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

        if not self.training:
            if global_step % 10 == 0:
                with torch.no_grad():            
                    n_rows, n_cols = 4, 4
                    img_grid = make_grid(self.input[0][0]).cpu().numpy()[-1]

                    bbox_gt = self.output['gt'][0][0].cpu().numpy().astype(int)
                    bbox_pred = self.output['preds'][0]['box'].cpu().numpy() # top bboxes

                    select_index = np.random.choice(np.arange(bbox_gt[2], bbox_gt[-1]), n_rows * n_cols, replace = False)
                    # sort the index
                    select_index.sort()
                    # select_index = np.random.choice(self.input[0].shape[-1], n_rows * n_cols, replace = False)
                    imgs_show = self.input[0][0][..., select_index].cpu().numpy()       

                    fig, axes = plt.subplots(n_rows, n_rows, figsize = (12, 12))
                    if not hasattr(axes, 'reshape'):
                        axes = [axes]
                    for i, ax in enumerate(axes.flatten()):
                        ax.axis('off'), ax.set_xticks([]), ax.set_yticks([])
                        ax.imshow(imgs_show[...,i], cmap='gray')
                        ax.add_patch(patches.Rectangle((bbox_gt[1], bbox_gt[0]), bbox_gt[4]-bbox_gt[1], bbox_gt[3]-bbox_gt[0], linewidth=1, edgecolor='r', facecolor='none'))
                        if bbox_pred.shape[0] != 0:
                            ax.add_patch(patches.Rectangle((bbox_pred[0][1], bbox_pred[0][0]), bbox_pred[0][4]-bbox_pred[0][1], bbox_pred[0][3]-bbox_pred[0][0], linewidth=1, edgecolor='g', facecolor='none'))
                            # ax.add_patch(patches.Rectangle((bbox_pred[1][1], bbox_pred[1][0]), bbox_pred[1][4]-bbox_pred[1][1], bbox_pred[1][3]-bbox_pred[1][0], linewidth=1, edgecolor='g', facecolor='none'))
                            # ax.add_patch(patches.Rectangle((bbox_pred[2][1], bbox_pred[2][0]), bbox_pred[2][4]-bbox_pred[2][1], bbox_pred[2][3]-bbox_pred[2][0], linewidth=1, edgecolor='g', facecolor='none'))
                        
                    fig.suptitle('{}_epoch{}.png'.format(mode, global_step))
                    # plt.savefig('tmp/check_test_plot/test_CoW.png'.format(global_step))

                    writer.add_figure(mode, fig, global_step)
                    if self.cfg.var.obj_operator.is_best:
                        writer.add_figure(f'{mode}_best', fig, global_step)
        
    