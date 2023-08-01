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
)

from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    SmartCacheDataset,
    PersistentDataset,
    load_decathlon_datalist,
)

class miccaimonaiDataset:
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
                        Orientationd(keys=["image", "label"], axcodes="IRA"),
                        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=plans['configurations'][model]['patch_size']),
                        # Spacingd(
                        #     keys=["image", "label"],
                        #     pixdim=plans['configurations'][model]['spacing'],
                        #     mode=("bilinear", "nearest"),
                        # ),
                        EnsureTyped(keys=["image", "label"], track_meta=False),
                        RandCropByPosNegLabeld(
                            keys=["image", "label"],
                            label_key="label",
                            spatial_size=plans['configurations'][model]['patch_size'],
                            pos=1,
                            neg=1,
                            num_samples=num_samples,
                        ),
                        # RandFlipd(
                        #     keys=["image", "label"],
                        #     spatial_axis=[0],
                        #     prob=0.10,
                        # ),
                        # RandFlipd(
                        #     keys=["image", "label"],
                        #     spatial_axis=[1],
                        #     prob=0.10,
                        # ),
                        # RandFlipd(
                        #     keys=["image", "label"],
                        #     spatial_axis=[2],
                        #     prob=0.10,
                        # ),
                        # RandRotate90d(
                        #     keys=["image", "label"],
                        #     prob=0.10,
                        #     max_k=3,
                        #     spatial_axes=(1,2),
                        # ),
                        # RandShiftIntensityd(
                        #     keys=["image"],
                        #     offsets=0.10,
                        #     prob=0.50,
                        # ),
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
                        Orientationd(keys=["image", "label"], axcodes="IRA"),
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