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

#check existance of samples and labels
def check_img_exist(db, dblist, icafe_dir):
    for cat in dblist:
        for pipickle in dblist[cat]:
            pi = pipickle.split('/')[-1]
            tif_name = icafe_dir + '/' + db + '/' + pipickle + '/TH_' + pi + '.npy'
            label_name = icafe_dir + '/' + db + '/' + pipickle + '/TH_' + pi + 'd.npy'
            if not os.path.exists(tif_name):
                raise FileNotFoundError('Not exist ' + tif_name)
            if not os.path.exists(label_name):
                raise FileNotFoundError('Not exist ' + label_name)
    return True

def loadImg(tif_name):
    img = np.load(tif_name)
    #img = img.astype(np.float32)
    return img

def loadLabel(label_name):
    label = np.load(label_name)
    #label = label.astype(np.uint8)
    return label

class lumenDataset(IterableDataset):
    def __init__(self, cfg, mode, path_dataset='./data/iSNAP'):
        super().__init__()
        self.cfg = cfg
        self.mode = mode

        # check if original images and segmentation mask images are available for each item
        paths_items_all = glob.glob(os.path.join(path_dataset, '*'))
        paths_items_avail = []
        for path_item in paths_items_all[:]: # Sometimes test on smaller dataset
            name_item = os.path.basename(path_item)
            path_img = os.path.join(path_item, f'TH_{name_item}.npy')
            path_seg = os.path.join(path_item, f'TH_{name_item}seg.npy')
            if not os.path.exists(path_img):
                print('original image does not exist: {path_img}')
                continue
            if not os.path.exists(path_seg):
                print('segmentation mask does not exist: {path_seg}')
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
        if self.mode == 'train':
            # if True:
            np.random.shuffle(self.paths_items)
            num = 0
            for path_item in self.paths_items:
                name_item = os.path.basename(path_item)
                path_img = os.path.join(path_item, f'TH_{name_item}.npy')
                path_seg = os.path.join(path_item, f'TH_{name_item}seg.npy')
                # path_seg = os.path.join(path_item, f'TH_{name_item}d.npy')
                img = np.load(path_img)
                seg = np.load(path_seg)
                assert img.shape == seg.shape
                
                ys, xs, zs = np.asarray(img > 0.05).nonzero() # For iSNAP, nearly all pixel values are higher than zero
                assert self.cfg.dataset.shape_patch < img.shape[0]
                # batch_patchs_img = torch.zeros(self.cfg.batch_size, 1, *[self.cfg.shape_patch] * 3)
                # batch_patchs_seg = torch.zeros(self.cfg.batch_size, 1, *[self.cfg.shape_patch] * 3)
                # batch_pos = np.zeros((self.cfg.batch_size, 3))
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
                    if np.max(patch_seg) == 0:
                        num -= 1
                        continue
                    patch_img = croppatch3d(img, y, x, z, *[self.cfg.dataset.shape_patch // 2] * 3)
                    # batch_pos[mi] = [x, y, z]

                    patch_seg = torch.tensor(patch_seg).float()
                    patch_img = torch.tensor(patch_img).unsqueeze(0).float()
                    patch_img = patch_img - patch_img.min()
                    patch_img = patch_img / patch_img.max()
                    yield patch_img, patch_seg
                if num == self.__len__() + 1:
                    break

        elif self.mode == 'val' or self.mode == 'test':
            for path_item in self.paths_items:
                name_item = os.path.basename(path_item)
                path_img = os.path.join(path_item, f'TH_{name_item}.npy')
                path_seg = os.path.join(path_item, f'TH_{name_item}seg.npy')
                img = np.load(path_img)
                seg = np.load(path_seg)

                ys, xs, zs = np.asarray(img > 0.05).nonzero() # For iSNAP, nearly all pixel values are higher than zero
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
