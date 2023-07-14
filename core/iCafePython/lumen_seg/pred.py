import numpy as np
from iCafePython.lumen_seg.img_utils import padImg
from iCafePython.lumen_seg.eval import DSC
import tifffile
from iCafePython.lumen_seg.dbloader import loadImg, loadLabel
import os
import keras
import copy
import matplotlib.pyplot as plt
from rich import print


def predCaseSliding(whole_pred_model, img, hps=22):
    #z of model
    half_z_size = int(whole_pred_model.input.shape[3]) // 2
    zmax = img.shape[2]
    stride = 2 * (half_z_size - hps)
    pred_img = np.repeat(np.zeros(img.shape)[..., None], 2, axis=-1)
    for si in range(zmax // stride - 1):
        ct = half_z_size + si * stride
        print('\rpred slices', ct - half_z_size, ct + half_z_size, end='')
        if ct + half_z_size < img.shape[2]:
            img_stack = img[None, :, :, ct - half_z_size:ct + half_z_size, None]
        else:
            #pad last slices
            img_stack_r = img[None, :, :, ct - half_z_size:ct + half_z_size, None]
            img_stack = np.zeros((1, img.shape[0], img.shape[1], 2 * half_z_size, 1))
            img_stack[0, :, :, :img_stack_r.shape[3], :] = img_stack_r
        if si == 0:
            # distance tranform map
            prob, dist = whole_pred_model.predict(img_stack)
            pred_img[:, :, :ct + half_z_size - hps] = np.concatenate([prob[0, :, :, :-hps, :], dist[0, :, :, :-hps, :]],
                                                                     axis=-1)
        elif si == zmax // stride - 2:
            zleft = zmax - (ct - half_z_size + hps)
            prob, dist = whole_pred_model.predict(img_stack)
            pred_img[:, :, ct - half_z_size + hps:] = np.concatenate(
                [prob[0, :, :, hps:hps + zleft, :], dist[0, :, :, hps:hps + zleft, :]], axis=-1)
        else:
            prob, dist = whole_pred_model.predict(img_stack)
            pred_img[:, :, ct - half_z_size + hps:ct + half_z_size - hps] = np.concatenate(
                [prob[0, :, :, hps:-hps:, :], dist[0, :, :, hps:-hps:, :]], axis=-1)

    return pred_img


def predCaseSlidingPatch(model, img, hps=22):
    img_pad = padImg(img, hps)
    #last channel 4 (first distance to centerline, 1:4 3d vector to centerline)
    pred_img = np.repeat(np.zeros(img_pad.shape)[..., None], 4, axis=-1)
    ct_img = np.zeros(img_pad.shape)
    for xi in range(1, img_pad.shape[0] // hps):
        print('\r', xi, img_pad.shape[0] // hps, end='')
        for yi in range(1, img_pad.shape[1] // hps):
            for zi in range(1, img_pad.shape[2] // hps):
                img_stack = img_pad[None, xi * hps - hps:xi * hps + hps, yi * hps - hps:yi * hps + hps,
                                    zi * hps - hps:zi * hps + hps, None]
                pred_img[xi * hps - hps:xi * hps + hps, yi * hps - hps:yi * hps + hps,
                         zi * hps - hps:zi * hps + hps] += model.predict(img_stack)[0, :, :, :]
                ct_img[xi * hps - hps:xi * hps + hps, yi * hps - hps:yi * hps + hps, zi * hps - hps:zi * hps + hps] += 1
    pred_img_norm = pred_img / ct_img
    return pred_img_norm[:img.shape[0], :img.shape[1], :img.shape[2]]


def prepareInputImg(base_model, whole_pred_model, tifimg, save_path=None):
    #tifimg is already normalized, but size not fit
    if tifimg.shape[0] != tifimg.shape[1]:
        max_axis = max(tifimg.shape[0], tifimg.shape[1])
        print('not square, pad to', max_axis)
        tif_norm_img = np.zeros((max_axis, max_axis, tifimg.shape[2]))
        tif_norm_img[max_axis // 2 - tifimg.shape[0] // 2:max_axis // 2 + tifimg.shape[0] // 2,
                     max_axis // 2 - tifimg.shape[1] // 2:max_axis // 2 + tifimg.shape[1] // 2] = tifimg
    else:
        tif_norm_img = copy.copy(tifimg)

    if save_path is not None:
        np.save(save_path, tif_norm_img.astype(np.float16))
    ori_tif_size = tif_norm_img.shape[0]
    if tif_norm_img.shape[0] < 512:
        #modify the input size for the whole pred model
        whole_pred_model = TargetSizeModel(base_model, whole_pred_model, tif_norm_img.shape[0], tif_norm_img.shape[1])
    else:
        #if size larger, crop edges
        tif_norm_img = tif_norm_img[tif_norm_img.shape[0] // 2 - 256:tif_norm_img.shape[0] // 2 + 256,
                                    tif_norm_img.shape[1] // 2 - 256:tif_norm_img.shape[1] // 2 + 256]
        whole_pred_model = TargetSizeModel(base_model, whole_pred_model, 512, 512)
    return tif_norm_img, whole_pred_model


def TargetSizeModel(base_model, whole_pred_model, height, width, depth=64):
    if whole_pred_model is not None and whole_pred_model.input.get_shape().as_list() == [None, height, width, depth, 1]:
        print('shape no change')
        return whole_pred_model
    input_layer = keras.layers.Input(shape=(height, width, depth, 1), name="input_new")
    pred_img = base_model(input_layer)
    whole_pred_model = keras.Model(input_layer, pred_img, name="whole_pred_model")
    #whole_pred_model.summary()
    return whole_pred_model


def predMetricList(base_model, whole_pred_model, case_list, path_data):
    metrics = []
    for pifolder in case_list:
        db = pifolder.split('/')[-2]
        pi = pifolder.split('/')[-1]
        tif_name = path_data + '/' + pi + '/TH_' + pi + '.npy'
        if not os.path.exists(tif_name):
            print('no exist', tif_name)
            continue
        label_name = path_data + '/' + pi + '/TH_' + pi + 'd.npy'
        if not os.path.exists(tif_name):
            print('skip', tif_name)
            continue
        img = loadImg(tif_name)

        norm_img, whole_pred_model = prepareInputImg(base_model, whole_pred_model, img)

        img_pred = predCaseSliding(whole_pred_model, norm_img)

        img_pred = restoreSize(img_pred, img.shape)

        label = loadLabel(label_name)

        #plt.imshow(np.max(img_pred[:,:,:,0],axis=2))
        #plt.show(block=False)

        cmetric = np.mean(DSC(img_pred[:, :, :, 0], label[:, :, :, 0] > 0))
        cdst = np.mean(abs(img_pred[:, :, :, 0] - label[:, :, :, 0]))
        print(' Dice:%.4f' % cmetric, 'Mean Dist:%.6f' % cdst)
        metrics.append(cmetric)
    return np.mean(metrics)


def savePred(img_pred, exp_path):
    img_pred = img_pred / np.max(img_pred) * 255
    img_pred = img_pred.astype(np.uint8)
    tifffile.imsave(exp_path, img_pred)


def restoreSize(img_pred, img_shape):
    if max(img_shape[0], img_shape[1]) > 512:
        img_pred_pad = np.zeros((max(img_shape[0], img_shape[1]), max(img_shape[0], img_shape[1]), img_shape[2], 2))
        img_pred_pad[img_pred_pad.shape[0] // 2 - 256:img_pred_pad.shape[0] // 2 + 256,
                     img_pred_pad.shape[1] // 2 - 256:img_pred_pad.shape[1] // 2 + 256] = img_pred
        img_pred = img_pred_pad

    if img_shape[0] != img_shape[1]:
        img_pred = img_pred[img_pred.shape[0] // 2 - img_shape[0] // 2:img_pred.shape[0] // 2 + img_shape[0] // 2,
                            img_pred.shape[1] // 2 - img_shape[1] // 2:img_pred.shape[1] // 2 + img_shape[1] // 2]
    return img_pred
