import glob
import os
import pickle
from PIL import Image
import numpy as np
import torch
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
    Spacingd,
    RandRotate90d,
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
        num_samples = 1 # temporary
        
        if mode == 'train':
            datalist = load_decathlon_datalist(datapath, True, "training")
            cls.transform = Compose(
                    [
                        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
                        ScaleIntensityRanged(
                            keys=["image"],
                            a_min=0,
                            a_max=1000,
                            b_min=0.0,
                            b_max=1.0,
                            clip=True,
                        ),
                        CropForegroundd(keys=["image", "label"], source_key="image"),
                        Orientationd(keys=["image", "label"], axcodes="RAS"),
                        Spacingd(
                            keys=["image", "label"],
                            pixdim=(0.5, 0.5, 0.8),
                            mode=("bilinear", "nearest"),
                        ),
                        EnsureTyped(keys=["image", "label"], device=device, track_meta=False),
                        RandCropByPosNegLabeld(
                            keys=["image", "label"],
                            label_key="label",
                            spatial_size=(96, 96, 96),
                            pos=1,
                            neg=0,
                            num_samples=num_samples,
                            image_key="image",
                            image_threshold=0,
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
                        ),
                        RandShiftIntensityd(
                            keys=["image"],
                            offsets=0.10,
                            prob=0.50,
                        ),
                    ]
                )
            
            # dataset = SmartCacheDataset(
            #                 data=datalist,
            #                 transform=cls.transform,
            #                 cache_num = 4,
            #             )
            dataset = PersistentDataset(
                            data=datalist,
                            transform=cls.transform,
                            cache_dir = './tmp/cache_dataset',
                        )            
            dataloader = ThreadDataLoader(dataset, num_workers=0, batch_size=cfg.exp.train.batch_size, shuffle=True)
            
        elif mode in ['val', 'test']:
            cls.transform = Compose(
                    [
                        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
                        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
                        CropForegroundd(keys=["image", "label"], source_key="image"),
                        Orientationd(keys=["image", "label"], axcodes="RAS"),
                        Spacingd(
                            keys=["image", "label"],
                            pixdim=(1.5, 1.5, 2.0),
                            mode=("bilinear", "nearest"),
                        ),
                        EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
                    ]
                )
            if mode == 'val':
                val_files = load_decathlon_datalist(datapath, True, "validation")
            elif mode == 'test':
                val_files = load_decathlon_datalist(datapath, True, "test")
            # dataset = SmartCacheDataset(data=val_files, transform=cls.transform, cache_num = 4)
            dataset = PersistentDataset(data=val_files, transform=cls.transform, cache_dir = './tmp/cache_dataset',)
            dataloader = ThreadDataLoader(dataset, num_workers=0, batch_size=cfg.exp[mode].batch_size)
        
        return dataset, dataloader