import keras
from keras.models import Model
from keras.layers import Input, Dense, Conv3D, RepeatVector, Reshape, MaxPooling3D, UpSampling3D, add, multiply, concatenate
from keras.utils import multi_gpu_model
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
import copy
import numpy as np

class lumenSeg():
    def __init__(self, model_name=None):
        if model_name is None:
            #for TOF segmentation, train from BRAVE project
            model_name = r'//DESKTOP2/Ftensorflow\LiChen\Y-net\LumenSeg\LumenSeg2-3\model021-0.03107.hdf5'
        print('load lumen seg model from',model_name)
        self.base_model = load_model(model_name,
                                     custom_objects={'dist_loss': dist_loss, 'kld': kld, 'relevant_mae': relevant_mae})
        self.whole_pred_model = None
        self.whole_pred_model = self.TargetSizeModel(512,512,64)
        self.ori_tif_size = 512

    def TargetSizeModel(self,height,width,depth=64):
        if self.whole_pred_model is not None and \
                self.whole_pred_model.input.get_shape().as_list() == [None, height,width,depth, 1]:
            print('shape no change')
            return self.whole_pred_model
        self.input_layer = keras.layers.Input(shape=(height,width, depth, 1), name="input_new")
        pred_img = self.base_model(self.input_layer)
        self.whole_pred_model = keras.Model(self.input_layer, pred_img, name="whole_pred_model")
        #whole_pred_model.summary()
        return self.whole_pred_model

    def prepareInputImg(self,tifimg,save_path=None,maxint=None):
        self.ori_tif_size = (tifimg.shape[0], tifimg.shape[1])
        self.max_axis = max(tifimg.shape[0], tifimg.shape[1])
        if tifimg.shape[0]!=tifimg.shape[1]:
            print('not square, pad to',self.max_axis)
            tif_norm_img = np.zeros((self.max_axis,self.max_axis,tifimg.shape[2]))
            tif_norm_img[tif_norm_img.shape[0]//2-tifimg.shape[0]//2:tif_norm_img.shape[0]//2+tifimg.shape[0]//2,
            tif_norm_img.shape[1] // 2 - tifimg.shape[1] // 2:tif_norm_img.shape[1] // 2 + (tifimg.shape[1] - tifimg.shape[1] // 2)] = tifimg
        else:
            tif_norm_img = copy.copy(tifimg)
        if maxint is not None:
            tif_norm_img[tif_norm_img > maxint] = maxint
        #1000 fixed number to allow range of signal around 0-1.xx
        #tif_norm_img = tif_norm_img / 1000
        if save_path is not None:
            np.save(save_path, tif_norm_img.astype(np.float16))

        if self.max_axis<512:
            #modify the input size for the whole pred model
            self.TargetSizeModel(tif_norm_img.shape[0],tif_norm_img.shape[1])
        else:
            #if size larger, crop from center
            tif_norm_img = tif_norm_img[tif_norm_img.shape[0] // 2 - 256:tif_norm_img.shape[0] // 2 + 256,
                       tif_norm_img.shape[1] // 2 - 256:tif_norm_img.shape[1] // 2 + 256]
            self.TargetSizeModel(512,512)
        return tif_norm_img

    def predCaseSliding(self, img, hps=22):
        # z of model
        half_z_size = int(self.whole_pred_model.input.shape[3]) // 2
        zmax = img.shape[2]
        stride = 2 * (half_z_size - hps)
        pred_img = np.repeat(np.zeros(img.shape,dtype=np.float16)[..., None], 2, axis=-1)
        for si in range(zmax // stride - 1):
            ct = half_z_size + si * stride
            print('\rpred slices', ct - half_z_size, ct + half_z_size, end='')
            if ct + half_z_size < img.shape[2]:
                img_stack = img[None, :, :, ct - half_z_size:ct + half_z_size, None]
            else:
                # pad last slices
                img_stack_r = img[None, :, :, ct - half_z_size:ct + half_z_size, None]
                img_stack = np.zeros((1, img.shape[0], img.shape[1], 2 * half_z_size, 1))
                img_stack[0, :, :, :img_stack_r.shape[3], :] = img_stack_r
            if si == 0:
                prob, dist = self.whole_pred_model.predict(img_stack)
                pred_img[:, :, :ct + half_z_size - hps] = \
                    np.concatenate([prob[0, :, :, :-hps, :], dist[0, :, :, :-hps, :]], axis=-1)
            elif si == zmax // stride - 2:
                zleft = zmax - (ct - half_z_size + hps)
                prob, dist = self.whole_pred_model.predict(img_stack)
                pred_img[:, :, ct - half_z_size + hps:] = \
                    np.concatenate([prob[0, :, :, hps:hps + zleft, :], dist[0, :, :, hps:hps + zleft, :]], axis=-1)
            else:
                prob, dist = self.whole_pred_model.predict(img_stack)
                pred_img[:, :, ct - half_z_size + hps:ct + half_z_size - hps] = \
                    np.concatenate([prob[0, :, :, hps:-hps:, :], dist[0, :, :, hps:-hps:, :]], axis=-1)
        if self.max_axis <= 512:
            pad_pred_img = pred_img
        else:
            pad_pred_img = np.zeros((self.max_axis,self.max_axis,pred_img.shape[2],2))
            pad_pred_img[self.max_axis//2-pred_img.shape[0]//2:self.max_axis//2+pred_img.shape[0]//2,
                self.max_axis//2-pred_img.shape[1]//2:self.max_axis//2+pred_img.shape[1]//2] = pred_img
        pad_img_ori = pad_pred_img[pad_pred_img.shape[0]//2-self.ori_tif_size[0]//2:pad_pred_img.shape[0]//2+self.ori_tif_size[0]//2,
                      pad_pred_img.shape[1]//2-self.ori_tif_size[1]//2:pad_pred_img.shape[1]//2+self.ori_tif_size[1]//2]
        img_prob = pad_img_ori[:, :, :, 0]
        img_dist = pad_img_ori[:, :, :, 1]
        return img_prob, img_dist

def lumenSeg3DUnet(width, height, depth, G=1, activationfun = 'relu', kernelinitfun = 'glorot_normal'):
    # conv transpose
    input_img = Input(shape=(width, height, depth, 1), name='patchimg')
    conv1 = Conv3D(16, (3, 3, 3), activation=activationfun, kernel_initializer=kernelinitfun, padding='same',
                   name='con1')(input_img)
    conv1_2 = Conv3D(16, (3, 3, 3), activation=activationfun, kernel_initializer=kernelinitfun, padding='same',
                   name='con12')(conv1)
    mp1 = MaxPooling3D((2, 2, 2), padding='same', name='pooling1')(conv1_2)
    conv2 = Conv3D(32, (3, 3, 3), activation=activationfun, kernel_initializer=kernelinitfun, padding='same',
                   name='con2')(mp1)
    conv2_2 = Conv3D(32, (3, 3, 3), activation=activationfun, kernel_initializer=kernelinitfun, padding='same',
                   name='con22')(conv2)
    mp2 = MaxPooling3D((2, 2, 2), padding='same', name='pooling2')(conv2_2)
    conv3 = Conv3D(64, (3, 3, 3), activation=activationfun, kernel_initializer=kernelinitfun, padding='same',
                   name='con3')(mp2)
    conv3_2 = Conv3D(64, (3, 3, 3), activation=activationfun, kernel_initializer=kernelinitfun, padding='same',
                   name='con32')(conv3)

    conv4 = Conv3D(32, (3, 3, 3), activation=activationfun, kernel_initializer=kernelinitfun, padding='same',
                   name='con4')(conv3_2)

    us1 = UpSampling3D((2, 2, 2))(conv4)
    concat1 = concatenate([conv2_2, us1], axis=4)
    conv5 = Conv3D(32, (3, 3, 3), activation=activationfun, kernel_initializer=kernelinitfun, padding='same',
                   name='con5')(concat1)
    conv5_2 = Conv3D(32, (3, 3, 3), activation=activationfun, kernel_initializer=kernelinitfun, padding='same',
                   name='con52')(conv5)
    us2 = UpSampling3D((2, 2, 2))(conv5_2)
    concat2 = concatenate([conv1_2, us2], axis=4)
    conv6 = Conv3D(16, (3, 3, 3), activation=activationfun, kernel_initializer=kernelinitfun, padding='same',
                   name='con6')(concat2)
    conv6_2 = Conv3D(16, (3, 3, 3), activation=activationfun, kernel_initializer=kernelinitfun, padding='same',
                   name='con62')(conv6)
    prob = Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same', name='prob')(conv6_2)
    dist = Conv3D(1, (3, 3, 3), padding='same', name='dist')(conv6_2)
    if G <= 1:
        print("[INFO] training with 1 GPU...")
        cnn = Model(inputs=input_img, outputs=[prob,dist])
    else:
        print("[INFO] training with {} GPUs...".format(G))
        with tf.device('/cpu:0'):
            # initialize the model
            s_cnn = Model(inputs=input_img, outputs=[prob,dist])
        # make the model parallel
        cnn = multi_gpu_model(s_cnn, gpus=G)

    # lr=0.01, momentum=0.9,nesterov =True
    #cnn.compile(optimizer='Adam', loss='binary_crossentropy')
    return cnn


def generic_masked_loss(mask, loss, weights=1, norm_by_mask=True, reg_weight=0, reg_penalty=K.abs):
    def _loss(y_true, y_pred):
        actual_loss = K.mean(mask * weights * loss(y_true, y_pred), axis=-1)
        norm_mask = (K.mean(mask) + K.epsilon()) if norm_by_mask else 1
        if reg_weight > 0:
            reg_loss = K.mean((1-mask) * reg_penalty(y_pred), axis=-1)
            return actual_loss / norm_mask + reg_weight * reg_loss
        else:
            return actual_loss / norm_mask
    return _loss

def masked_loss(mask, penalty, reg_weight, norm_by_mask):
    loss = lambda y_true, y_pred: penalty(y_true - y_pred)
    return generic_masked_loss(mask, loss, reg_weight=reg_weight, norm_by_mask=norm_by_mask)

# TODO: should we use norm_by_mask=True in the loss or only in a metric?
#       previous 2D behavior was norm_by_mask=False
#       same question for reg_weight? use 1e-4 (as in 3D) or 0 (as in 2D)?

def masked_loss_mae(mask, reg_weight=0, norm_by_mask=True):
    return masked_loss(mask, K.abs, reg_weight=reg_weight, norm_by_mask=norm_by_mask)

def masked_loss_mse(mask, reg_weight=0, norm_by_mask=True):
    return masked_loss(mask, K.square, reg_weight=reg_weight, norm_by_mask=norm_by_mask)

def masked_metric_mae(mask):
    def relevant_mae(y_true, y_pred):
        return masked_loss(mask, K.abs, reg_weight=0, norm_by_mask=True)(y_true, y_pred)
    return relevant_mae

def masked_metric_mse(mask):
    def relevant_mse(y_true, y_pred):
        return masked_loss(mask, K.square, reg_weight=0, norm_by_mask=True)(y_true, y_pred)
    return relevant_mse

def kld(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.mean(K.binary_crossentropy(y_true, y_pred) - K.binary_crossentropy(y_true, y_true), axis=-1)

def split_dist_true_mask(dist_true_mask):
    #tfx,tfy,tfz = tf.split(dist_true_mask, num_or_size_splits=[1, 1, 1], axis=-1)
    tfm = tf.math.reduce_max(dist_true_mask,axis=-1,keepdims=True)
    mask = tf.math.greater(tfm, tf.constant([0],dtype=tf.float32))
    mask = tf.cast(mask,tf.float32)
    return mask, dist_true_mask

def dist_loss(dist_true_mask, dist_pred):
    dist_mask, dist_true = split_dist_true_mask(dist_true_mask)
    masked_dist_loss = masked_loss_mse
    return masked_dist_loss(dist_mask, reg_weight=True)(dist_true, dist_pred)

def relevant_mae(dist_true_mask, dist_pred):
    dist_mask, dist_true = split_dist_true_mask(dist_true_mask)
    return masked_metric_mae(dist_mask)(dist_true, dist_pred)

def relevant_mse(dist_true_mask, dist_pred):
    dist_mask, dist_true = split_dist_true_mask(dist_true_mask)
    return masked_metric_mse(dist_mask)(dist_true, dist_pred)
