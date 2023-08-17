import glob
import os
import pickle
from PIL import Image
import numpy as np
import torch
import random
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
    SpatialCropd,
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

import numpy as np
import random
from monai.transforms import Cropd, SpatialCrop
from typing import Sequence, Optional

class WeightedSpatialCropd(Cropd):
    
    def __init__(
        self,
        keys,
        roi_size: Optional[Sequence[int]] = None,
        allow_missing_keys: bool = False,
        num_patches: int = 1,
        lazy: bool = False,
    ) -> None:
        self.roi_size = roi_size
        self.roi_center = None  # Initialize roi_center to None; it will be updated in __call__
        self.num_patches = num_patches

        super().__init__(keys, cropper=None, allow_missing_keys=allow_missing_keys, lazy=lazy)

    def __call__(self, data):
        patches = []
        
        pdb.set_trace()
        # First patch that includes the stenosis region
        self.set_stenosis_center(data)
        patches.append(self.create_patch(data))

        # Generate random patches
        for _ in range(1, self.num_patches):
            self.set_random_center(data)
            patches.append(self.create_patch(data))

        return patches

    def set_stenosis_center(self, data):
        # Get the stenosis center from the data
        bbox_string, spacing_string = data['extra'].split(',')

        if bbox_string == '[-1]':
            shape = data['image'].shape
            stenosis_center = [random.randint(0, shape[i] - self.roi_size[i]) for i in range(len(shape))]
        else:
            # transform the bbox
            bbox = np.array([int(i) for i in bbox_string[1:-1].split(' ')])
            stenosis_center = [random.randint(bbox[0], bbox[1]), random.randint(bbox[2], bbox[3]), random.randint(bbox[4], bbox[5])]

        self.roi_center = stenosis_center

    def set_random_center(self, data):
        shape = data['image'].shape
        # Get the lower and upper bounds for each dimension to keep the patch within the image boundary
        lower_bounds = [self.roi_size[i] // 2 for i in range(len(shape))]
        upper_bounds = [shape[i] - self.roi_size[i] // 2 for i in range(len(shape))]

        # Choose the random center such that the entire patch is within the image boundary
        random_center = [random.randint(lower_bounds[i], upper_bounds[i]) for i in range(len(shape))]
        self.roi_center = random_center

    def create_patch(self, data):
        # Update the SpatialCrop with the current roi_center
        self.cropper = SpatialCrop(roi_center=self.roi_center, roi_size=self.roi_size)

        # Apply cropping
        cropped_data = {}
        for key in self.keys:
            cropped_data[key] = self.cropper(data[key])

        return cropped_data


# Only for validation and test
class ProcessExtraKey(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        for key in self.keys:
            extra = data[key]
            # process the 'extra' data here
            processed_data = self.process(extra)

            # save the processed data as new keys
            data[f"meta"] = processed_data
            del data[key]
        return data

    def process(self, extra):

        # Here the extra_info is in '[x0 x1 y0 y1 z0 z1],[spacing0 spacing1 spacing2]' format and the stenosis bounding box can also be in '-1' format
        # I want to store these information in a dictionary with keys 'bbox' and 'spacing'
        bbox_string, spacing_string = extra.split(',')
        bbox = np.array([int(i) for i in bbox_string[1:-1].split(' ')])
        spacing = np.array([float(i) for i in spacing_string[1:-1].split(' ')])
        processed_data = {'s_bbox': bbox, 'spacing': spacing}

        return processed_data


class smrastenosisDataset:
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
                    LoadImaged(keys=["image", "label"], reader="NumpyReader",
                               ensure_channel_first=True, image_only=False),
                    # LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=False),
                    NormalizeIntensityd(
                        keys=["image"],
                        subtrahend=dataset_fingerprint["foreground_intensity_properties_per_channel"][fold]["mean"],
                        divisor=dataset_fingerprint["foreground_intensity_properties_per_channel"][fold]["std"],),
                    # CropForegroundd(keys=["image", "label"], source_key="image"),
                    Orientationd(keys=["image", "label"], axcodes="SRA"),
                    ResizeWithPadOrCropd(keys=["image", "label"],
                                         spatial_size=plans['configurations'][model]['patch_size']),
                    # Spacingd(
                    #     keys=["image", "label"],
                    #     pixdim=plans['configurations'][model]['spacing'],
                    #     mode=("bilinear", "nearest"),
                    # ),
                    EnsureTyped(keys=["image", "label"], track_meta=False),

                    WeightedSpatialCropd(
                        keys=["image", "label", "extra"],
                        roi_size=plans['configurations'][model]['patch_size'],
                        num_patches=num_samples,
                        allow_missing_keys=True,
                    ),
                    # pay attention to the transformation since your stenosis location should also change
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
                cache_num=cfg.dataset.num_cache_train,
            )
            pdb.set_trace()

            dataloader = ThreadDataLoader(dataset, num_workers=0, batch_size=cfg.exp.train.batch_size, shuffle=True)

        # If load numpy file, you should use orientation SRA
        # If load nii file, you should use orientation ILP
        elif mode == 'val':
            cls.transform = Compose(
                [
                    LoadImaged(keys=["image", "label"], reader="NumpyReader",
                               ensure_channel_first=True, image_only=False),
                    # LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=False),
                    NormalizeIntensityd(
                        keys=["image"],
                        subtrahend=dataset_fingerprint["foreground_intensity_properties_per_channel"][fold]["mean"],
                        divisor=dataset_fingerprint["foreground_intensity_properties_per_channel"][fold]["std"],),
                    # CropForegroundd(keys=["image", "label"], source_key="image"),
                    # ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=plans['configurations'][model]['patch_size']),
                    # Orientationd(keys=["image", "label"], axcodes="ILP"),
                    Orientationd(keys=["image", "label"], axcodes="SRA"),
                    # Spacingd(
                    #     keys=["image", "label"],
                    #     pixdim=plans['configurations'][model]['spacing'],
                    #     mode=("bilinear", "nearest"),
                    # ),
                    ProcessExtraKey(keys=["extra"]),
                    EnsureTyped(keys=["image", "label"], track_meta=False),
                ]
            )
            val_files = load_decathlon_datalist(datapath, True, "validation")

            dataset = SmartCacheDataset(data=val_files, transform=cls.transform, cache_num=cfg.dataset.num_cache_val)
            dataloader = ThreadDataLoader(dataset, num_workers=0, batch_size=cfg.exp[mode].batch_size)

        elif mode == 'test':
            cls.transform = Compose(
                [
                    LoadImaged(keys=["image", "label"], reader="NumpyReader",
                               ensure_channel_first=True, image_only=False),
                    # LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=False),
                    NormalizeIntensityd(
                        keys=["image"],
                        subtrahend=dataset_fingerprint["foreground_intensity_properties_per_channel"][fold]["mean"],
                        divisor=dataset_fingerprint["foreground_intensity_properties_per_channel"][fold]["std"],),
                    # Orientationd(keys=["image", "label"], axcodes="ILP"),
                    Orientationd(keys=["image", "label"], axcodes="SRA"),
                    ProcessExtraKey(keys=["extra"]),
                    EnsureTyped(keys=["image", "label"], track_meta=False),
                ]
            )
            test_files = load_decathlon_datalist(datapath, True, "test")

            dataset = SmartCacheDataset(data=test_files, transform=cls.transform, cache_num=cfg.dataset.num_cache_test)
            dataloader = ThreadDataLoader(dataset, num_workers=0, batch_size=cfg.exp[mode].batch_size)

        return dataset, dataloader
