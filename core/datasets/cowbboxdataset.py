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
import pdb

from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    SmartCacheDataset,
    PersistentDataset,
    load_decathlon_datalist,
    DataLoader,
    Dataset,
)
from monai.transforms import ScaleIntensityRanged
from monai.data.utils import no_collation

from core.utils.bbox_transform import generate_detection_train_transform, generate_detection_val_transform, generate_detection_inference_transform

intensity_transform = ScaleIntensityRanged(
        keys=["image"],
        a_min=-1024,
        a_max=300.0,
        b_min=0.0,
        b_max=1.0,
        clip=True,
    )

class cowbboxDataset:
    def __new__(cls, cfg, mode):
        
        datapath = cfg.dataset.path
        
        device = cfg.var.obj_operator.device
        num_samples = cfg.dataset.num_samples

        if mode == 'train':

            train_transforms = generate_detection_train_transform(
                "image",
                "box",
                "label",
                cfg.dataset.gt_box_mode,
                intensity_transform,
                cfg.dataset.patch_size,
                cfg.exp.train.batch_size,
                affine_lps_to_ras=False,
                amp=True,
                seg_key = "seg",
            )

            datalist = load_decathlon_datalist(datapath, True, "training")
            
            dataset = Dataset(
                data=datalist,
                transform=train_transforms,
            )

            dataloader = DataLoader(
                dataset,
                batch_size=cfg.dataset.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=torch.cuda.is_available(),
                collate_fn=no_collation,
                persistent_workers=True,
            )
            
        elif mode in ['val', 'test']:

            val_transforms = generate_detection_val_transform(
                "image",
                "box",
                "label",
                cfg.dataset.gt_box_mode,
                intensity_transform,
                affine_lps_to_ras=True,
                amp=True,
                seg_key = "seg",
            )
            
            if mode == 'val':
                val_files = load_decathlon_datalist(datapath, True, "validation")
            elif mode == 'test':
                val_files = load_decathlon_datalist(datapath, True, "test")
            
            dataset = Dataset(
                data=val_files,
                transform=val_transforms,
            )

            dataloader = DataLoader(
                dataset,
                batch_size=cfg.exp[mode].batch_size,
                num_workers=2,
                pin_memory=torch.cuda.is_available(),
                collate_fn=no_collation,
                persistent_workers=True,
            )
        
        return dataset, dataloader