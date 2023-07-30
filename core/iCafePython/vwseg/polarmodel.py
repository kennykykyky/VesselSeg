import tensorflow as tf
from keras import backend as K
import keras
from keras.models import load_model
from .vwseg_utils import polar_pred_cont_cst, toctbd, plotct, DSC, diffmap
import cv2
from ..utils.img_utils import topolar, tocart
import numpy as np
import matplotlib.pyplot as plt

class PolarBase:
    @property
    def cartbd(self):
        if self._cartbd is None:
            if self.polarbd is None:
                raise ValueError('Polar not seg')
            contourin, contourout = toctbd(self.polarbd, self.cfg['width'], self.cfg['width'])
            self._cartbd = (contourin, contourout)
        return self._cartbd

    @property
    def cartseg(self):
        if self._carseg is None:
            contourin, contourout = self.cartbd
            self._carseg = plotct(self.cfg['width'] * 2, contourin, contourout)
        return self._carseg

    def resetImg(self):
        self.polarpred, self.polarbd, self.polarsd = None, None, None
        self._cartbd = None
        self._carseg = None
        self.polarpatch = None

    def toPolar(self,cart_img):
        #cart img in original size, height,width,depth
        polar_img = np.zeros((self.cfg['height'],self.cfg['width'],cart_img.shape[2]))
        for di in range(cart_img.shape[2]):
            cart_img_enlarge = cv2.resize(cart_img[:,:,di],(0,0),fx=4*self.cfg['SCALE'],fy=4*self.cfg['SCALE'])
            polar_img[:,:,di] = topolar(cart_img_enlarge / np.max(cart_img_enlarge), self.cfg['width'], self.cfg['height'])
        return polar_img

class PolarReg(PolarBase):
    def __init__(self,model_path=None):
        cfg = {}
        #cfg['modelname'] = 'ICAReg1-1'
        cfg['modelname'] = 'ICAReg1-3'
        #model 1
        #cfg['modelpath'] = '//Desktop4/Dtensorflow/LiChen/VW/ICAReg/ICAReg1-1/ModelEpo065-0.23777-0.75202.hdf5'
        #model 2, realm renew, first 10 cases used as training
        cfg['modelpath'] = '//Desktop4/Dtensorflow/LiChen/VW/ICAReg/ICAReg1-8/ModelEpo122-0.04013-0.77108.hdf5'
        cfg['width'] = 256
        cfg['height'] = 256
        cfg['patchheight'] = 256
        cfg['depth'] = 3
        cfg['channel'] = 1
        cfg['SCALE'] = 1
        cfg['vw_pred_type'] = 'reg'
        self.cfg = cfg
        #load model
        self.loadModel(model_path)
        self.resetImg()


    def setModel(self,model_path):
        self.cfg['modelname'] = model_path.replace('\\','/')[-2]
        self.cfg['modelpath'] = model_path

    def loadModel(self,model_path=None):
        def polarloss(y_true, y_pred):
            dmax = K.sum(tf.math.maximum(y_true, y_pred))
            dmin = K.sum(tf.math.minimum(y_true, y_pred))
            return K.log(tf.math.divide(dmax, dmin))
        if model_path is not None:
            self.setModel(model_path)
        print('loading model from',self.cfg['modelpath'])
        self.model = load_model(self.cfg['modelpath'], custom_objects={'polarloss': polarloss})
        self.feat_model = None

    def predict(self,polarimg):
        #polarimg shape of
        if polarimg.shape!=(self.cfg['height'],self.cfg['width'],self.cfg['depth']):
            raise ValueError('Image input size mismatch with model',polarimg.shape,(self.cfg['height'],self.cfg['width'],self.cfg['depth']))
        self.resetImg()
        self.polarpatch = polarimg
        #add channel dimension
        polarimg = polarimg[:, :, :, None]
        self.polarbd, self.polarsd = polar_pred_cont_cst(polarimg, self.cfg, self.model)
        return self.polarbd, self.polarsd

    def polar_plot(self, polar_plot_name=None, cart_label=None):
        polar_patch = self.polarpatch
        cart_patch = tocart(self.polarpatch)
        if cart_label is not None:
            polar_label = topolar(cart_label)
        else:
            polar_label = None
        polarbd = self.polarbd
        cart_seg = self.cartseg

        fz = 20
        fig = plt.figure(figsize=(20, 8))
        plt.subplot(2, 5, 1)
        plt.title('Cartesian Patch', fontsize=fz)
        plt.imshow(cart_patch, cmap='gray')

        plt.subplot(2, 5, 2)
        if cart_label is not None:
            plt.title('Prediction \nDSC:%.3f' % (DSC(cart_seg, cart_label)), fontsize=fz)
            plt.imshow(diffmap(cart_seg, cart_label), cmap='gray')
        else:
            plt.title('Prediction', fontsize=fz)
            plt.imshow(cart_seg, cmap='gray')

        if cart_label is not None:
            plt.subplot(2, 5, 3)
            plt.title('Cartesian Label', fontsize=fz)
            plt.imshow(cart_label, cmap='gray')

        '''if cart_unet is not None:
            plt.subplot(2, 5, 4)
            plt.title('U-Net Prediction \nDSC:%.3f' % (DSC(cart_unet, cart_label)), fontsize=fz)
            plt.imshow(diffmap(cart_unet, cart_label), cmap='gray')
        if cart_mask is not None:
            plt.subplot(2, 5, 5)
            plt.title('Mask-RCNN Prediction \nDSC:%.3f' % (DSC(cart_mask, cart_label)), fontsize=fz)
            plt.imshow(diffmap(cart_mask, cart_label), cmap='gray')'''



        plt.subplot(2, 5, 6)
        plt.title('Polar Patch', fontsize=fz)
        plt.imshow(polar_patch, cmap='gray')

        plt.subplot(2, 5, 7)
        plt.title('Polar Prediction', fontsize=fz)
        plt.xlim([0, 256])
        plt.ylim([polarbd.shape[0], 0])
        plt.plot(polarbd[::4, 0], np.arange(0, polarbd.shape[0], 4), 'o', markersize=2, label='Lumen')
        plt.plot(polarbd[::4, 1], np.arange(0, polarbd.shape[0], 4), 'o', markersize=2, label='Wall')
        plt.legend()

        if polar_label is not None:
            plt.subplot(2, 5, 8)
            plt.title('Polar Label', fontsize=fz)
            plt.imshow(polar_label, cmap='gray')

        # fig.tight_layout()
        if polar_plot_name is not None:
            plt.savefig(polar_plot_name)
        else:
            plt.show()
        plt.close()

    #pred long height patches with sliding window
    def polar_pred_long_patch(self,polar_img):
        polar_pred = np.zeros((polar_img.shape[0], 2))
        polar_ct = np.zeros((polar_img.shape[0]))
        stride_polar = 32
        for stridei in range((polar_img.shape[0] - self.cfg['patchheight']) // stride_polar):
            patch_s = stridei * stride_polar
            patch_e = stridei * stride_polar + self.cfg['patchheight']
            polar_pred_unit = self.predict(polar_img[patch_s:patch_e])[0]
            polar_pred[patch_s:patch_e] += polar_pred_unit
            polar_ct[patch_s:patch_e] += 1

        # last batch
        patch_s = polar_img.shape[0] - self.cfg['patchheight']
        patch_e = polar_img.shape[0]
        polar_pred_unit = self.predict(polar_img[patch_s:patch_e])[0]
        polar_pred[patch_s:patch_e] += polar_pred_unit
        polar_ct[patch_s:patch_e] += 1

        polarbd_all = np.zeros((polar_img.shape[0], 2))
        for pti in range(polar_pred.shape[0]):
            polarbd_all[pti] = polar_pred[pti] / polar_ct[pti]

        return polarbd_all

    def featPred(self,polar_patch):
        if self.feat_model is None:
            self.feat_model = keras.Model(inputs=self.model.input,
                                          outputs=self.model.get_layer('aux_fx1').output)
        features = self.feat_model.predict(polar_patch[None,:,:,:,None])
        return features[0]

    def featPredStack(self,polar_patch):
        if self.feat_model is None:
            self.feat_model = keras.Model(inputs=self.model.input,
                                          outputs=self.model.get_layer('aux_fx1').output)
        polar_stack = np.zeros((polar_patch.shape[2], self.cfg['width'], self.cfg['height'],self.cfg['depth'], 1))
        for i in range(polar_patch.shape[2]):
            if i==0:
                polar_stack[i, :, :, 0, 0] = polar_patch[:, :, 0]
                polar_stack[i, :, :, 1, 0] = polar_patch[:, :, 0]
                polar_stack[i, :, :, 2, 0] = polar_patch[:, :, 1]
            elif i == polar_patch.shape[2]-1:
                polar_stack[i, :, :, 0, 0] = polar_patch[:, :, i-1]
                polar_stack[i, :, :, 1, 0] = polar_patch[:, :, i]
                polar_stack[i, :, :, 2, 0] = polar_patch[:, :, i]
            else:
                polar_stack[i, :, :, :, 0] = polar_patch[:, :, i-1:i+2]
        features = self.feat_model.predict(polar_stack)
        return features

class PolarSegReg(PolarBase):
    def __init__(self):
        pass
    def TODO(self):
        pass
