import os
import copy
import numpy as np
# from skimage import io
import pdb
from PIL import Image
import tifffile
from interpolation.splines import UCGrid, CGrid, nodes
from interpolation.splines import eval_linear
import matplotlib.pyplot as pyplot
from .point3d import Point3D
from .utils.img_utils import expandImg, rtTransform
import pydicom
import glob
from .utils.img_utils import resample3d
from .interp.interp_utils import getCSPos
from .utils.img_utils import paint_dist, paint_dist_transform
# from iCafePython import point3d

#list available images
def listAvailImgs(self):
    avail_list = []
    for src in ['', 'v', 's', 'b', 'i', 'n', 's.whole'] + ['S1' + str(x).zfill(2) for x in range(15)
                                                           ] + ['S1' + str(x).zfill(2) + '_ori' for x in range(15)]:
        tif_file_name = os.path.join(self.path, 'TH_' + self.filename_solo + src + '.tif')
        if os.path.exists(tif_file_name):
            avail_list.append(src)
            if src == '':
                avail_list.append('o')
    return avail_list


def _saveImgFile(path, img, format=np.int16):
    tifimg = np.swapaxes(img, 0, 2)
    if format is not None:
        tifimg = tifimg.astype(format)
    tifffile.imwrite(path, tifimg)
    print('img save to', path)


def saveImg(self, src, img, format=np.int16, path=None):
    if path is None:
        path = os.path.join(self.path, 'TH_' + self.filename_solo + src + '.tif')

    if src == 'v':
        print('crop v img to', self.xml.cropregion)
        img = img[self.xml.cropregion[0]:self.xml.cropregion[2], self.xml.cropregion[1]:self.xml.cropregion[3]]
    _saveImgFile(path, img, format)


def saveImgAs(self, target_foler, src, format=np.int16):
    path = os.path.join(target_foler, 'TH_' + self.filename_solo + src + '.tif')
    if src not in self.I:
        if src in self.listAvailImgs():
            self.loadImg(src)
        else:
            raise ValueError('no exist I src')
    self._saveImgFile(path, self.I[src], format)


#save as to new target folder
def saveImgs(self, target_foler):
    for src in self.listAvailImgs():
        self.saveImgAs(target_foler, src)


#load individual image
def _loadImgFile(self, path, src='o', expandimg=False):
    if not os.path.exists(path):
        print('Not exist', path)
        return
    else:
        tifimg = np.swapaxes(tifffile.imread(path), 0, 2)
        # tifimg = np.swapaxes(io.imread(path), 0, 2)
        if expandimg:
            tifimg = expandImg(tifimg, self.xml.cropregion, self.tifimg.shape)
        #additional operations for specific srcs
        if src == 'o':
            self._tifimg = tifimg
            self._SM = tifimg.shape[0]
            self._SN = tifimg.shape[1]
            self._SZ = tifimg.shape[2]
        elif src == 's':
            tifimg = self.tifimg / np.max(self.tifimg)
        elif src == 'i':
            icafeitxt = os.path.join(self.path, 'TH_' + self.filename_solo + 'i.txt')
            itxt = {}
            with open(icafeitxt, 'r') as fp:
                for line in fp:
                    items = line.split('\t')
                    itxt[int(items[0])] = int(items[1][:-1])
            idtovesmap = np.array(list(itxt.values()))
            self.vesimg = idtovesmap[tifimg]
        elif src[-4:] == '_ori':
            self.posRTMat[src] = self.xml.readSeqRTM(src[:-4])

        self.I[src] = copy.copy(tifimg)
        if type(self.I[src][0, 0, 0]) == np.float16:
            self.I[src] = self.I[src].astype(np.float)
        # self.Iint[src] = UCGrid((0, tifimg.shape[0] - 1, tifimg.shape[0]), (0, tifimg.shape[1] - 1, tifimg.shape[1]),
        #                         (0, tifimg.shape[2] - 1, tifimg.shape[2]))


# load images from self.I dict
def loadImgs(self):
    for src in list(self.I.keys()):
        self.loadImg(src)


def loadImg(self, src, img=None, path=None):
    if img is not None:
        if type(img[0, 0, 0]) == np.float16:
            self.I[src] = img.astype(np.float)
        else:
            self.I[src] = img
        self.Iint[src] = UCGrid((0, img.shape[0] - 1, img.shape[0]), (0, img.shape[1] - 1, img.shape[1]),
                                (0, img.shape[2] - 1, img.shape[2]))
    else:
        assert self.filename_solo is not None
        if path is None:
            if src == 'o':
                tif_file_name = os.path.join(self.path, 'TH_' + self.filename_solo + '.tif')
            else:
                tif_file_name = os.path.join(self.path, 'TH_' + self.filename_solo + src + '.tif')
        else:
            tif_file_name = path
        if src == 'v':
            # vessel image is cropped to smaller size, need to restore to original size
            expand_to_original_size = True
        else:
            expand_to_original_size = False
        self._loadImgFile(tif_file_name, src, expand_to_original_size)
        if self.I[src] is None:
            del self.I[src]
            return None
    return self.I[src]


def loadVWI(self):
    if 'S101_ori' in self.listAvailImgs():
        print('Load S101_ori')
        self.loadImg('S101_ori')
        vw_seq1 = 'S101_ori'
    elif 'S101' in self.listAvailImgs():
        print('Load S101')
        self.loadImg('S101')
        vw_seq1 = 'S101'
    else:
        raise FileNotFoundError('S101 Not exist')
    return vw_seq1


def getInt(self, pos, src='o'):
    if src not in self.I.keys():
        if src not in self.listAvailImgs():
            raise ValueError('src ', src, 'not exist among', self.listAvailImgs())
        else:
            self.loadImg(src)
            print('load src', src)
    if type(pos) == list:
        pos = Point3D(pos)
    if src in self.posRTMat:
        #print('previous',pos)
        pos = rtTransform(pos, self.posRTMat[src])
        #print('after', pos)
    if self.rzratio != 1:
        pos.z /= self.rzratio

    if pos.outOfBound(self.I[src].shape):
        return 0
    else:
        if src == 'i':
            return self.I[src][tuple(pos.intlst())]
        else:
            return eval_linear(self.Iint[src], self.I[src], np.array((pos.x, pos.y, pos.z)))


def getBoxInt(self, pos, hps=1, src='o'):
    if src not in self.I.keys():
        raise ValueError('src not exist', src)
    int_box = np.zeros((2 * hps + 1, 2 * hps + 1, 2 * hps + 1))
    for xi in range(-hps, hps + 1):
        for yi in range(-hps, hps + 1):
            for zi in range(-hps, hps + 1):
                int_box[xi + hps, yi + hps, zi + hps] = self.getInt(pos + Point3D(xi, yi, zi), src=src)
    return np.mean(int_box)


def getSphereInt(self, pos, hps=1, src='o'):
    if src not in self.I.keys():
        raise ValueError('src not exist', src)
    int_box = []
    for xi in range(-hps, hps + 1):
        for yi in range(-hps, hps + 1):
            for zi in range(-hps, hps + 1):
                if Point3D(xi, yi, zi).dist(Point3D(0, 0, 0)) <= hps:
                    int_box.append(self.getInt(pos + Point3D(xi, yi, zi), src=src))
    return np.mean(int_box)


#display a slice of original image
def displaySlice(self, slicei, src='o'):
    pyplot.imshow(np.transpose(self.I[src][:, :, slicei]), cmap='gray')
    pyplot.show()


# extract one slice from image, if ori src, interp
def extractSlice(self, slicei, src='o', axis=2):
    if src not in self.I:
        raise ValueError('Image src not loaded')
    if slicei < 0 or slicei >= self.I[src].shape[2]:
        raise ValueError('Slicei over range of', self.I[src].shape[2])
    if src[-4:] == '_ori':
        if axis == 2:
            img_slice = np.zeros((self.SM, self.SN))
            for ri in range(self.SM):
                for ci in range(self.SN):
                    img_slice[ri, ci] = self.getInt(Point3D(ri, ci, slicei), src)
            return img_slice
        else:
            raise ValueError('TODO')
    else:
        if axis == 0:
            return self.I[src][slicei]
        elif axis == 1:
            return self.I[src][:, slicei]
        elif axis == 2:
            return self.I[src][:, :, slicei]


def getIntensityAlongSnake(self, snake, src='o', int_pos=False):
    intensity_along_snake = []
    for pti in range(snake.NP):
        cpos = snake[pti].pos
        if int_pos:
            cpos.toIntPos()
        intensity_along_snake.append(self.getInt(cpos, src))
    return intensity_along_snake


def getIntensityRaySnake(self, snake, ri=2, angle_step=30, src='o'):
    intensity_ray_snake = []
    for pti in range(snake.NP):
        pos = snake[pti].pos
        rad = snake[pti].rad
        norm = snake.getNorm(pti)
        ray_pos_angles = []
        for angle in range(0, 360, angle_step):
            u = rad * ri * np.cos(angle / 180 * np.pi)
            v = rad * ri * np.sin(angle / 180 * np.pi)
            ray_pos = getCSPos(norm, pos, u, v)
            ray_pos_angles.append(self.getInt(ray_pos, src))
        intensity_ray_snake.append(ray_pos_angles)
    return intensity_ray_snake


#generate snake map where each vessel (no anatomical labels) will be painted in the map, saved in l.tif
def genSnakeMap(self, save_label=False):
    label_img = np.zeros((self.SM, self.SN, self.SZ), dtype=np.uint16)
    for snakeid in range(self.snakelist.NSnakes):
        #print('\rpainting snake',snakeid,end='')
        for pti in range(self.snakelist[snakeid].NP):
            pos = self.snakelist[snakeid][pti].pos
            rad = self.snakelist[snakeid][pti].rad
            # paint within radius at pos position
            paint_dist(label_img, pos, rad, snakeid + 1)
    self.I['l'] = label_img
    if save_label is True:
        label_filename = self.path + '/TH_' + self.filename_solo + 'l.tif'
        tifffile.imwrite(label_filename, np.swapaxes(label_img, 0, 2))
        print('save label image to', label_filename)
    return label_img


#generate artery map where each artery (with anatomical labels) will be painted in the map, saved in i.tif and i.txt
def genArtMap(self, save_label=False):
    ves_mapping = {0: 0}
    label_img = np.zeros((self.SM, self.SN, self.SZ), dtype=np.uint16)
    for snakeid in range(self.vessnakelist.NSnakes):
        #print('\rpainting snake',snakeid,end='')
        for pti in range(self.vessnakelist[snakeid].NP):
            pos = self.vessnakelist[snakeid][pti].pos
            rad = self.vessnakelist[snakeid][pti].rad
            # paint within radius at pos position
            paint_dist(label_img, pos, rad, snakeid + 1)
            ves_mapping[snakeid + 1] = self.vessnakelist[snakeid].type

    self.I['i'] = label_img
    if save_label is True:
        label_filename = self.path + '/TH_' + self.filename_solo + 'i.tif'
        tifffile.imwrite(label_filename, np.swapaxes(label_img, 0, 2))
        print('save label image to', label_filename)

        #save i.txt
        label_txt_filename = self.path + '/TH_' + self.filename_solo + 'i.txt'
        with open(label_txt_filename, 'w') as fp:
            for ves_key in ves_mapping:
                fp.write('%d\t%d\n' % (ves_key, ves_mapping[ves_key]))

    return label_img


def paintDistTransform(self, snakelist, path=None):
    # generate distance transformed labeled image
    img_fill = np.zeros((self.tifimg.shape))
    # first channel distance to centerline (normalized to 0-1), last three channels absolute 3d distance to centerline
    img_fill = np.repeat(img_fill[:, :, :, None], 4, axis=3)

    for snakeid in range(snakelist.NSnakes):
        print('\r', snakeid, '/', snakelist.NSnakes, end='')
        unit_snake = snakelist[snakeid].resampleSnake(1)
        for pti in unit_snake:
            ct = pti.pos
            rad = pti.rad
            paint_dist_transform(img_fill, ct, rad)

    # # export distance img
    # self.saveImg('d', img_fill[:, :, :, 0], np.uint8)

    # export distance img and distance vector map
    if path is None:
        dmap_npy_file = self.getPath('d.npy')
    else:
        dmap_npy_file = path
    np.save(dmap_npy_file, img_fill.astype(np.float16))


def searchDCM(self, path=None):
    if path is None:
        path = self.datapath
    if not os.path.exists(path):
        print('no dcms in folder')
        return
    return glob.glob(path + '/*.dcm')


def createFromVTS(self, dcm_path, mra_seq='S104'):
    dcm_files_mra = glob.glob(dcm_path + '/*' + mra_seq + 'I*.dcm')
    dcm_files_mra.sort(key=lambda x: int(os.path.basename(x).split(mra_seq + 'I')[-1][:-4]))
    print('creating tif from ', len(dcm_files_mra), 'matching dcms')
    dcm = pydicom.read_file(dcm_files_mra[0])
    #in VTS the slice thickness is different
    slice_thickness = float(dcm.SliceThickness) #float(dcm.SpacingBetweenSlices) +
    print('slice_thickness', slice_thickness)
    return self.createFromDCM(dcm_files_mra, slice_thickness=slice_thickness)


def createFromDCM(self, dcm_files, pixel_spacing=None, slice_thickness=None):
    dcm = pydicom.read_file(dcm_files[0])
    img = np.zeros((dcm.Rows, dcm.Columns, len(dcm_files)))

    for slicei in range(len(dcm_files)):
        dcm_filename = dcm_files[slicei]
        dcm = pydicom.read_file(dcm_filename)
        img[:, :, slicei] = dcm.pixel_array

    img = np.swapaxes(img, 0, 1)

    if pixel_spacing is None:
        if 'PixelSpacing' in dcm:
            pixel_spacing = dcm.PixelSpacing[0]
        else:
            print('pixel spacing not available')
            pixel_spacing = 1

    if slice_thickness is None:
        if 'SpacingBetweenSlices' in dcm:
            spacing_b_slices = float(dcm.SpacingBetweenSlices)
            slice_thickness = spacing_b_slices
            print('found slice_thickness', slice_thickness, 'from SpacingBetweenSlices')
        elif 'SliceThickness' in dcm:
            slice_thickness = float(dcm.SliceThickness)
            print('found slice_thickness', slice_thickness, 'from SliceThickness')
        else:
            print('thickness info not available')
            slice_thickness = pixel_spacing

    return self.createFromImg(img, pixel_spacing, slice_thickness)


def create_from_npy(self, path_npy, pixel_spacing=1, slice_thickness=1):
    array = np.load(path_npy)
    return self.createFromImg(array, pixel_spacing, slice_thickness)


def createFromImg(self, img, pixel_spacing=None, slice_thickness=None, resamp=1):
    #img is 3D array with dimension of height width and depth
    print('ori img size', img.shape)

    if slice_thickness is not None and pixel_spacing is not None:
        ztoxy_spacing_scale = slice_thickness / pixel_spacing
    else:
        ztoxy_spacing_scale = 1

    if resamp or not os.path.exists(self.getPath('.tif')):
        self.saveImg('', img)
        if ztoxy_spacing_scale != 1:
            # resize in z axis to be iso
            print('resize z axis to iso')
            resample3d(self.getPath('.tif'), ztoxy_spacing_scale, 1)

        if not os.path.exists(self.datapath):
            os.makedirs(self.datapath)

        if pixel_spacing is not None:
            self.xml.setResolution(pixel_spacing)
        self.xml.writexml()


from .norm.nyul import nyulFromNormPerc


def normImg(self, ref_perc, method='nyul'):
    if method == 'nyul':
        return nyulFromNormPerc(self.tifimg, ref_perc)
    else:
        raise ValueError('not defined')


def imComputeInitForegroundModel(self):
    # consider all the seed points as initial foreground
    if len(self.seeds) == 0:
        raise ValueError('no seeds available')
    I_seed_int = [self.getInt(si) for si in self.seeds]
    Iv_seed_int = [self.getInt(si, 'v') for si in self.seeds]
    self.u1, self.sigma1, self.uv1, self.sigmav1 = np.mean(I_seed_int), np.std(I_seed_int), np.mean(
        Iv_seed_int), np.std(Iv_seed_int)
    return self.u1, self.sigma1, self.uv1, self.sigmav1


def imComputeInitBackgroundModel(self):
    self.u2, self.sigma2, self.uv2, self.sigmav2 = np.mean(self.I['o']), np.std(self.I['o']), \
        np.mean(self.I['v']), np.std(self.I['v'])
    return self.u2, self.sigma2, self.uv2, self.sigmav2