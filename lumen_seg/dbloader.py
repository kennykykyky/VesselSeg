import os
#import torch
import tifffile
import numpy as np
from iCafePython.lumen_seg.img_utils import croppatch3d
from keras.utils.np_utils import to_categorical
from rich import print


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


#lumen segmentation dataset
#class LSDataset(torch.utils.data.IterableDataset):
class LSDataset():
    def __init__(self, cfg, path_data, pilist, auglist=[]):
        self.cfg = cfg
        self.path_data = path_data
        self.hps = self.cfg.patch_size // 2
        self.pilist = pilist
        self.auglist = auglist
        self.output_pos = False
        self.batch_size = 16

    def __len__(self):
        return len(self.pilist)

    def __iter__(self):
        while 1:
            np.random.shuffle(self.pilist)
            for pifolder in self.pilist:
                pi = os.path.basename(pifolder)
                tif_name = self.path_data + '/' + pi + '/TH_' + pi + '.npy'
                if not os.path.exists(tif_name):
                    print('no exist', tif_name)
                    continue
                label_name = self.path_data + '/' + pi + '/TH_' + pi + 'd.npy'
                img = loadImg(tif_name)
                label = loadLabel(label_name)
                assert img.shape == label.shape[:-1]
                ylist, xlist, zlist = np.where(label[:, :, :, 0] > 0)
                #print('valid voxels',len(xlist))
                patch_size = self.cfg.patch_size
                sz = img.shape[0]
                assert patch_size < sz
                mi = 0
                batch_img_patch = np.zeros((self.batch_size, 2 * self.hps, 2 * self.hps, 2 * self.hps, 1))
                batch_label_patch = np.zeros((self.batch_size, 2 * self.hps, 2 * self.hps, 2 * self.hps, 4))
                batch_pos = np.zeros((self.batch_size, 3))
                for ri in range(len(ylist) // 500):
                    rid = np.random.randint(len(xlist))
                    ctx = max(
                        0, min(label.shape[1] - 1, xlist[rid] + np.random.randint(-patch_size // 2, patch_size // 2)))
                    cty = max(
                        0, min(label.shape[0] - 1, ylist[rid] + np.random.randint(-patch_size // 2, patch_size // 2)))
                    ctz = max(
                        0, min(label.shape[2] - 1, zlist[rid] + np.random.randint(-patch_size // 2, patch_size // 2)))
                    label_patch = croppatch3d(label, cty, ctx, ctz, self.hps, self.hps, self.hps)
                    if np.max(label_patch) == 0:
                        continue
                    img_patch = croppatch3d(img, cty, ctx, ctz, self.hps, self.hps, self.hps)

                    batch_img_patch[mi] = img_patch[:, :, :, None]
                    batch_label_patch[mi] = label_patch[:, :, :, :]
                    batch_pos[mi] = [ctx, cty, ctz]

                    mi += 1

                    if mi == self.batch_size // 4:
                        batch_img_patch[mi:mi * 2] = batch_img_patch[:mi, ::-1]
                        batch_label_patch[mi:mi * 2] = batch_label_patch[:mi, ::-1]
                        batch_img_patch[mi * 2:mi * 3] = batch_img_patch[:mi, :, ::-1]
                        batch_label_patch[mi * 2:mi * 3] = batch_label_patch[:mi, :, ::-1]
                        batch_img_patch[mi * 3:mi * 4] = batch_img_patch[:mi, :, :, ::-1]
                        batch_label_patch[mi * 3:mi * 4] = batch_label_patch[:mi, :, :, ::-1]

                        batch_pos[mi:mi * 2] = batch_pos[:mi]
                        batch_pos[mi * 2:mi * 3] = batch_pos[:mi]
                        batch_pos[mi * 3:mi * 4] = batch_pos[:mi]
                        if self.output_pos:
                            yield batch_img_patch, batch_label_patch, batch_pos
                        else:
                            #categorical_labels = to_categorical(batch_label_patch[:, :, :, 1], num_classes=7)
                            categorical_labels = batch_label_patch[..., 0:1]
                            yield batch_img_patch, [batch_label_patch[..., 0:1] > 0, categorical_labels]

                        mi = 0


#db loader for training. Along snakes with random offset, extract label from i.tif(artery map)
# Snakes not labeled as arteries has zeros in i.tif
#lumen segmentation from snake
class LSSDataset():
    def __init__(self, cfg, path_data, pilist, auglist=[]):
        self.cfg = cfg
        self.path_data = path_data
        self.hps = self.cfg.patch_size // 2
        self.pilist = pilist
        self.auglist = auglist
        self.output_pos = False
        self.batch_size = 16
        self.max_pts_per_case = 500
        self.raw_vesname = 'tracing_raw_ves_TH_'
        self.vesname = 'tracing_ves_TH_'

    def __len__(self):
        return len(self.pilist)

    def __iter__(self):
        while 1:
            np.random.shuffle(self.pilist)
            for pifolder in self.pilist:
                pi = os.path.basename(pifolder)
                tif_name = self.path_data + '/' + pi + '/TH_' + pi + '.npy'
                label_name = self.path_data + '/' + pi + '/TH_' + pi + 'd.npy'
                if not os.path.exists(label_name):
                    print('no exist', label_name)
                    continue
                img = loadImg(tif_name)
                label = loadLabel(label_name)
                if self.cfg.weight_raw_ves > 0:
                    snake_pts_raw = self.readVesPts(self.path_data + '/' + pi + '/' + self.raw_vesname + pi + '.swc')
                else:
                    snake_pts_raw = []
                if self.cfg.weight_ves>0 and \
                        os.path.exists((self.path_data + '/' + pi + '/'+self.vesname + pi + '.swc')):
                    snake_pts_ves = self.readVesPts(self.path_data + '/' + pi + '/' + self.vesname + pi + '.swc')
                else:
                    snake_pts_ves = []

                rand_pts = []
                for ri in range(int(self.cfg.weight_rand*(len(snake_pts_raw)*self.cfg.weight_raw_ves+\
                                                      len(snake_pts_ves) *self.cfg.weight_ves))):
                    rand_pts.append([
                        np.random.randint(0, label.shape[0] - 1),
                        np.random.randint(0, label.shape[1] - 1),
                        np.random.randint(0, label.shape[2] - 1)
                    ])
                snake_pts = snake_pts_raw * self.cfg.weight_raw_ves + snake_pts_ves * self.cfg.weight_ves + rand_pts
                patch_size = self.cfg.patch_size
                sz = img.shape[0]
                assert patch_size < sz
                mi = 0
                batch_img_patch = np.zeros((self.batch_size, 2 * self.hps, 2 * self.hps, 2 * self.hps, 1))
                batch_label_patch = np.zeros((self.batch_size, 2 * self.hps, 2 * self.hps, 2 * self.hps, 4))
                batch_pos = np.zeros((self.batch_size, 3))
                #print('snake pts',len(snake_pts)//(self.raw_ves_weight+self.ves_weight))

                for ri in range(
                        min(self.max_pts_per_case,
                            len(snake_pts) // (self.cfg.weight_raw_ves + self.cfg.weight_ves))):
                    rid = np.random.randint(len(snake_pts))
                    ctx = max(
                        0,
                        min(label.shape[1] - 1,
                            snake_pts[rid][1] + np.random.randint(-patch_size // 2, patch_size // 2)))
                    cty = max(
                        0,
                        min(label.shape[0] - 1,
                            snake_pts[rid][0] + np.random.randint(-patch_size // 2, patch_size // 2)))
                    ctz = max(
                        0,
                        min(label.shape[2] - 1,
                            snake_pts[rid][2] + np.random.randint(-patch_size // 2, patch_size // 2)))
                    label_patch = croppatch3d(label, cty, ctx, ctz, self.hps, self.hps, self.hps)
                    img_patch = croppatch3d(img, cty, ctx, ctz, self.hps, self.hps, self.hps)

                    batch_img_patch[mi] = img_patch[:, :, :, None]
                    batch_label_patch[mi] = label_patch[:, :, :, :]
                    batch_pos[mi] = [ctx, cty, ctz]

                    mi += 1

                    if mi == self.batch_size // 4:
                        batch_img_patch[mi:mi * 2] = batch_img_patch[:mi, ::-1]
                        batch_label_patch[mi:mi * 2] = batch_label_patch[:mi, ::-1]
                        batch_img_patch[mi * 2:mi * 3] = batch_img_patch[:mi, :, ::-1]
                        batch_label_patch[mi * 2:mi * 3] = batch_label_patch[:mi, :, ::-1]
                        batch_img_patch[mi * 3:mi * 4] = batch_img_patch[:mi, :, :, ::-1]
                        batch_label_patch[mi * 3:mi * 4] = batch_label_patch[:mi, :, :, ::-1]

                        batch_pos[mi:mi * 2] = batch_pos[:mi]
                        batch_pos[mi * 2:mi * 3] = batch_pos[:mi]
                        batch_pos[mi * 3:mi * 4] = batch_pos[:mi]
                        if self.output_pos:
                            yield batch_img_patch, batch_label_patch, batch_pos
                        else:
                            # categorical_labels = to_categorical(batch_label_patch[:, :, :, 1], num_classes=7)
                            categorical_labels = batch_label_patch[..., 0:1]
                            yield batch_img_patch, [batch_label_patch[..., 0:1] > 0, categorical_labels]

                        mi = 0

    def readVesPts(self, ves_filename):
        snake_pts = []
        with open(ves_filename, 'r') as fp:
            for line in fp:
                ct = line.split(' ')
                cpos = [float(i) for i in ct[2:5]]
                #crad = float(ct[5])
                snake_pts.append(cpos)
        return snake_pts
