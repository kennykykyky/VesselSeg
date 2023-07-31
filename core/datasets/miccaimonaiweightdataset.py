import glob
import os
import pickle
from PIL import Image
import numpy as np
import torch
import json
from torch.utils.data import Dataset, IterableDataset
import pdb
from core.utils.img import croppatch3d
import nibabel as nib
import torch.distributed as dist
from monai.transforms import Randomizable

from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    NormalizeIntensityd,
    Spacingd,
    RandRotate90d,
    ResizeWithPadOrCropd,
    EnsureTyped,
    Randomizable,
    Transform,
    MapTransform,
)

from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    SmartCacheDataset,
    PersistentDataset,
    load_decathlon_datalist,
)

class AddNf(MapTransform):
    def __init__(self, keys, nf_values):
        super().__init__(keys)
        self.nf_values = nf_values

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key]['nf'] = self.nf_values[key]  # add 'nf' field to each item
        return d

class WeightedRandCropByPosNegLabeld(Randomizable, Transform):
    def __init__(self, keys, label_key, spatial_size, pos, neg, num_samples, image_key=None):
        self.cropper = RandCropByPosNegLabeld(keys, label_key, spatial_size, pos, neg, num_samples, image_key, allow_missing_keys=True)
        
    def randomize(self, data=None):
        self.cropper.randomize(data)

    def __call__(self, data):
        d = dict(data)
        extra_info = d['extra']
        # delete 'extra' field from each item
        del d['extra']
        croppeds = self.cropper(d)
        # Assuming 'extra' in d is the location of interest
        weights = self.calculate_weight(croppeds, extra_info)
        # add weight field to each item in the list
        for cropped, weight in zip(croppeds, weights):
                cropped['weight'] = weight
        return croppeds

    def calculate_weight(self, patches, bbox_string):
        weights = []
        for patch in patches:
            # check if the value of bbox is a string '-1' and then it will return weight 1
            if bbox_string == '-1':
                weight = 1.0
            else:
                # transform the bbox in a string format '[x0, x1, y0, y1, z0, z1]' to a numpy array of ints [x0, x1, y0, y1, z0, z1]
                bbox = np.array([int(i) for i in bbox_string[1:-1].split(',')])

                # Get the spatial size of the patch
                patch_size = patch['image'].shape[1:]  # assuming patch['image'] is a 4D tensor with shape (C, H, W, D)

                # Get the coordinates of the patch's origin in world space
                patch_origin = patch['image_meta_dict']['affine'] @ np.array([0, 0, 0, 1])

                # Calculate the coordinates of the patch's corners in world space
                patch_bbox = [patch_origin[0], patch_origin[0] + patch_size[0], 
                            patch_origin[1], patch_origin[1] + patch_size[1], 
                            patch_origin[2], patch_origin[2] + patch_size[2]]

                # Check if the patch and bounding box intersect
                if (bbox[0] < patch_bbox[1] and bbox[1] > patch_bbox[0] and
                    bbox[2] < patch_bbox[3] and bbox[3] > patch_bbox[2] and
                    bbox[4] < patch_bbox[5] and bbox[5] > patch_bbox[4]):
                    # The patch intersects with the bounding box, assign a higher weight
                    weight = 1.0
                else:
                    # The patch does not intersect with the bounding box, assign a lower weight
                    weight = 0.5

            weights.append(weight)

        return weights

class miccaimonaiweightDataset:
    def __new__(cls, cfg, mode):
        
        datapath = cfg.dataset.path
        
        device = cfg.var.obj_operator.device
        num_samples = cfg.dataset.num_samples

        if cfg.exp.nnunet_result.path:
            # load json file to datset_fingerprint by joining cfg.exp.nnunet_result_path and dataset_fingerprint.json
            with open(os.path.join(cfg.exp.nnunet_result.path, 'dataset_fingerprint.json'), 'r') as f:
                dataset_fingerprint = json.load(f)
            with open(os.path.join(cfg.exp.nnunet_result.path, 'plans.json'), 'r') as f:
                plans = json.load(f)
        
        fold = str(cfg.exp.nnunet_result.fold)
        model = cfg.exp.nnunet_result.model

        if mode == 'train':
            datalist = load_decathlon_datalist(datapath, True, "training")
            cls.transform = Compose(
                    [
                        LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=False),
                        NormalizeIntensityd(
                            keys=["image"],
                            subtrahend=dataset_fingerprint["foreground_intensity_properties_per_channel"][fold]["mean"],
                            divisor=dataset_fingerprint["foreground_intensity_properties_per_channel"][fold]["std"],),
                        # CropForegroundd(keys=["image", "label"], source_key="image"),
                        Orientationd(keys=["image", "label"], axcodes="SRA"),
                        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=plans['configurations'][model]['patch_size']),
                        # Spacingd(
                        #     keys=["image", "label"],
                        #     pixdim=plans['configurations'][model]['spacing'],
                        #     mode=("bilinear", "nearest"),
                        # ),
                        EnsureTyped(keys=["image", "label"], track_meta=False),
                        WeightedRandCropByPosNegLabeld(
                            keys=["image", "label", "extra"],
                            label_key="label",
                            spatial_size=plans['configurations'][model]['patch_size'],
                            pos=1,
                            neg=1,
                            num_samples=num_samples,
                        ),
                        RandFlipd(
                            keys=["image", "label"],
                            spatial_axis=[0],
                            prob=0.10,
                        ),
                        RandFlipd(
                            keys=["image", "label"],
                            spatial_axis=[1],
                            prob=0.10,
                        ),
                        RandFlipd(
                            keys=["image", "label"],
                            spatial_axis=[2],
                            prob=0.10,
                        ),
                        RandRotate90d(
                            keys=["image", "label"],
                            prob=0.10,
                            max_k=3,
                            spatial_axes=(1,2),
                        ),
                        RandShiftIntensityd(
                            keys=["image"],
                            offsets=0.10,
                            prob=0.50,
                        ),
                    ]
                )
            
            dataset = SmartCacheDataset(
                            data=datalist,
                            transform=cls.transform,
                            cache_num = cfg.dataset.num_cache_train,
                        )

            dataloader = ThreadDataLoader(dataset, num_workers=0, batch_size=cfg.exp.train.batch_size, shuffle=True)
            
        elif mode in ['val', 'test']:
            cls.transform = Compose(
                    [
                        LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=False),
                        NormalizeIntensityd(
                            keys=["image"],
                            subtrahend=dataset_fingerprint["foreground_intensity_properties_per_channel"][fold]["mean"],
                            divisor=dataset_fingerprint["foreground_intensity_properties_per_channel"][fold]["std"],),
                        # CropForegroundd(keys=["image", "label"], source_key="image"),
                        # ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=plans['configurations'][model]['patch_size']),
                        Orientationd(keys=["image", "label"], axcodes="SRA"),
                        # Spacingd(
                        #     keys=["image", "label"],
                        #     pixdim=plans['configurations'][model]['spacing'],
                        #     mode=("bilinear", "nearest"),
                        # ),
                        EnsureTyped(keys=["image", "label"], track_meta=False),
                    ]
                )
            if mode == 'val':
                val_files = load_decathlon_datalist(datapath, True, "validation")
            elif mode == 'test':
                val_files = load_decathlon_datalist(datapath, True, "test")
            dataset = SmartCacheDataset(data=val_files, transform=cls.transform, cache_num = cfg.dataset.num_cache_val)
            dataloader = ThreadDataLoader(dataset, num_workers=0, batch_size=cfg.exp[mode].batch_size)
        
        return dataset, dataloader