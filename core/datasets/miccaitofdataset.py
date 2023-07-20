import glob
import os
import pickle
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
import torchvision
import torchvision.transforms as transforms
import pdb
from core.utils.img import croppatch3d
import nibabel as nib
import torch.distributed as dist

class miccaitofDataset(IterableDataset):
    def __init__(self, cfg, mode, path_dataset='./data/CAS2023_training'):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        
        self.n_yielded = 0
        
        self.path_dataset = path_dataset

        # check if original images and segmentation mask images are available for each item
        paths_items_all = glob.glob(os.path.join(path_dataset, '*/*.nii.gz'))
        paths_items_avail = []
        for path_item in paths_items_all[:]: # Sometimes test on smaller dataset
            name_item = os.path.basename(path_item)
            path_img = os.path.join(path_dataset, f'data/{name_item}')
            path_seg = os.path.join(path_dataset, f'mask/{name_item}')
            if not os.path.exists(path_img):
                print('original image does not exist: {}'.format(path_img))
                continue
            if not os.path.exists(path_seg):
                print('segmentation mask does not exist: {}'.format(path_seg))
                continue
            paths_items_avail.append(path_item)
        paths_items_avail.sort()

        length = len(paths_items_avail)
        if mode == 'train':
            self.paths_items = paths_items_avail[:int(length * 0.8)]
        elif mode == 'val':
            self.paths_items = paths_items_avail[int(length * 0.8): int(length * 0.9)]
        elif mode == 'test':
            self.paths_items = paths_items_avail[int(length * 0.9):]
        else:
            raise NotImplementedError

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths_items)

    def __iter__(self):
        if self.cfg.var.is_parallel:
            
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        
        
        if self.mode == 'train':
            # if True:
            np.random.shuffle(self.paths_items)
            num = 0
            for path_item in self.paths_items:
                name_item = os.path.basename(path_item)
                path_img = os.path.join(self.path_dataset, f'data/{name_item}')
                path_seg = os.path.join(self.path_dataset, f'mask/{name_item}')
                img = nib.load(path_img)
                seg = nib.load(path_seg)
                
                img = img.get_fdata()
                seg = seg.get_fdata()
                assert img.shape == seg.shape
                
                ys, xs, zs = np.asarray(seg == 1).nonzero() # For iSNAP, nearly all pixel values are higher than zero
                assert self.cfg.dataset.shape_patch < img.shape[0]

                for _ in range(len(xs) // 500):
                    num += 1
                    if num == self.__len__() + 1:
                        break
                    i = np.random.randint(len(xs))
                    x = xs[i] + np.random.randint(-self.cfg.dataset.shape_patch // 2, self.cfg.dataset.shape_patch // 2)
                    y = ys[i] + np.random.randint(-self.cfg.dataset.shape_patch // 2, self.cfg.dataset.shape_patch // 2)
                    z = zs[i] + np.random.randint(-self.cfg.dataset.shape_patch // 2, self.cfg.dataset.shape_patch // 2)
                    x = np.clip(x, 0, seg.shape[1] - 1)
                    y = np.clip(y, 0, seg.shape[0] - 1)
                    z = np.clip(z, 0, seg.shape[2] - 1)
                    patch_seg = croppatch3d(seg, y, x, z, *[self.cfg.dataset.shape_patch // 2] * 3)
                    # if no lumen region, abort current patch
                    patch_img = croppatch3d(img, y, x, z, *[self.cfg.dataset.shape_patch // 2] * 3)
                    # batch_pos[mi] = [x, y, z]

                    patch_seg = torch.tensor(patch_seg).float()
                    patch_img = torch.tensor(patch_img).unsqueeze(0).float()
                    patch_img = patch_img - patch_img.min()
                    patch_img = patch_img / patch_img.max()
                    
                    yield patch_img, patch_seg
                    # if self.cfg.var.is_parallel:
                    #     if (self.n_yielded + rank) % world_size == 0:
                    #         self.n_yielded += 1
                    #         yield patch_img, patch_seg
                    # else:
                    #     yield patch_img, patch_seg
                        
                if num == self.__len__() + 1:
                    break

        elif self.mode == 'val' or self.mode == 'test':
            for path_item in self.paths_items:
                name_item = os.path.basename(path_item)
                path_img = os.path.join(self.path_dataset, f'data/{name_item}')
                path_seg = os.path.join(self.path_dataset, f'mask/{name_item}')
                img = nib.load(path_img)
                seg = nib.load(path_seg)
                
                img = img.get_fdata()
                seg = seg.get_fdata()

                ys, xs, zs = np.asarray(seg == 1).nonzero()
                i = np.random.randint(len(xs))
                x = xs[i] + np.random.randint(-self.cfg.dataset.shape_patch // 2, self.cfg.dataset.shape_patch // 2)
                y = ys[i] + np.random.randint(-self.cfg.dataset.shape_patch // 2, self.cfg.dataset.shape_patch // 2)
                z = zs[i] + np.random.randint(-self.cfg.dataset.shape_patch // 2, self.cfg.dataset.shape_patch // 2)
                x = np.clip(x, 0, seg.shape[1] - 1)
                y = np.clip(y, 0, seg.shape[0] - 1)
                z = np.clip(z, 0, seg.shape[2] - 1)
                img = croppatch3d(img, y, x, z, *[self.cfg.dataset.shape_patch // 2] * 3)
                seg = croppatch3d(seg, y, x, z, *[self.cfg.dataset.shape_patch // 2] * 3)

                seg = torch.tensor(seg).float()
                img = torch.tensor(img).unsqueeze(0).float()
                img = img - img.min()
                img = img / img.max()
                img = img[None, ...]
                seg = seg[None, ...]
                yield img, seg

    @staticmethod
    def add_configs(parser):
        parser.add_argument('--num_channels_input', default=1, type=int)
        parser.add_argument('--dim', default=3, type=int)
        parser.add_argument('--mods', default=['tof'])
        parser.add_argument('--shape_patch', default=48, type=int)

        return parser
                        
    def to_device(self, batch, device):
        imgs, segs = batch
        imgs, segs = imgs.to(device), segs.to(device)
        if self.mode != 'train':
            assert imgs.shape[0] == 1
            imgs, segs = imgs[0], segs[0]
        return {'imgs': imgs, 'segs': segs}
