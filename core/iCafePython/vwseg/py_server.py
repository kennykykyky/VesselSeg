import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import tensorflow as tf
import keras
import datetime

import tensorflow as tf
from keras import backend as K
import keras
from keras.models import load_model
from polar_utils import polar_pred_cont_cst, toctbd, plotct, DSC, diffmap
import cv2
from UTL import croppatch, topolar, tocart

import numpy as np
import matplotlib.pyplot as plt
import xmlclass.etree.ElementTree as ET

xsize = 80
ysize = 80
cnnRes = 0.28
SERVER_DIR = 'D:/tensorflow/LiChen/Server/'
UPLOAD_FOLDER = SERVER_DIR + 'upload/VW/'
SAVE_FOLDER = SERVER_DIR + 'download/VW/'
DOWNLOAD_LINK = 'http://128.208.221.39:5000/download/VW/'
VWCALFEAT = 'D:/gce/getwtd.exe'
print('TF loaded!')


def processrequest(request):
    file = request.files['image']
    print(request.form)

    px = 0
    if 'px' in request.form:
        px = int(round(float(request.form['px'])))

    py = 0
    if 'py' in request.form:
        py = int(round(float(request.form['py'])))

    pixelSize = 0
    if 'pixelSize' in request.form:
        pixelSize = float(request.form['pixelSize'])

    probimg = 0
    if 'probimg' in request.form:
        probimg = int(float(request.form['probimg']))

    scaleres = 1
    if 'res' in request.form:
        scaleres = float(request.form['res']) / cnnRes
        if scaleres <= 0:
            scaleres = 1

    ENLARGEMENT = 4
    if 'enl' in request.form:
        ENLARGEMENT = int(request.form['enl'])
    # para = request.files['para']
    # print(para)
    f = os.path.join(UPLOAD_FOLDER, file.filename)
    print(f)
    # add your custom code to check that the uploaded file is a valid image and not a malicious file (out-of-scope for this post)
    file.save(f)

    # whether to calculate feature
    calfeat = 0
    if 'calfeat' in request.form:
        calfeat = int(request.form['calfeat'])

    return polarpredimg(file.filename, px, py, scaleres, ENLARGEMENT, probimg, calfeat)


def errorxml(txt):
    QVAS_Image = ET.Element('QVAS_Image')
    QVAS_Success = ET.SubElement(QVAS_Image, 'QVAS_Success')
    QVAS_Success.text = txt
    xml = ET.tostring(QVAS_Image)
    return xml


def polarpredimg(filename, xcenter, ycenter, scaleres, ENLARGEMENT, probimg, calfeat):
    DEBUG = 0

    starttime = datetime.datetime.now()
    polarmodel = PolarReg()

    test_srcr = cv2.imread(os.path.join(UPLOAD_FOLDER, filename))

    print('Int range', np.min(test_srcr), np.max(test_srcr))
    if len(test_srcr.shape) == 3:
        test_srcr = test_srcr[:, :, 0]
    if scaleres != 1:
        test_src = cv2.resize(test_srcr, (0, 0), fx=scaleres, fy=scaleres)
    else:
        test_src = test_srcr

    xcenter = int(round(xcenter * scaleres))
    ycenter = int(round(ycenter * scaleres))

    hps = 64 // polarmodel.cfg['SCALE']

    if xcenter < hps or xcenter >= test_src.shape[1] - hps or xcenter == 0:
        xcenter = test_src.shape[1] // 2
    if ycenter < hps or ycenter >= test_src.shape[0] - hps or ycenter == 0:
        ycenter = test_src.shape[0] // 2

    cs_width = hps * 2
    cart_stack = np.zeros((cs_width, cs_width, polarmodel.cfg['depth']))
    for si in range(polarmodel.cfg['depth']):
        cartpatch = croppatch(test_src, cty=ycenter, ctx=xcenter, sheight=hps, swidth=hps)
        cart_stack[:, :, si] = cartpatch / np.max(cartpatch)
    if DEBUG:
        plt.imshow(cart_stack)
        plt.show()
    polar_stack = polarmodel.toPolar(cart_stack)
    endalltime = datetime.datetime.now()
    elaspsealltime1 = endalltime - starttime
    print('topolar polarpatch', elaspsealltime1)

    polarbd, polarsd = polarmodel.predict(polar_stack)

    endalltime = datetime.datetime.now()
    elaspsealltime2 = endalltime - starttime
    print('Prediction done', elaspsealltime2)

    polarbdrestore = polarbd  # / polarmodel.cfg['width'] * (ysize//2*ENLARGEMENT)

    cartseg = polarmodel.cartseg

    segbin = cv2.resize(cartseg, (0, 0), fx=1 / ENLARGEMENT, fy=1 / ENLARGEMENT)

    segsavepath = SAVE_FOLDER + 'seg_' + filename
    segbin = croppatch(segbin, -1, -1, ysize // 2, xsize // 2)
    segbin = segbin / np.max(segbin) * 255
    cv2.imwrite(segsavepath, segbin)

    endalltime = datetime.datetime.now()
    elaspsealltime5 = endalltime - starttime
    print('output results', elaspsealltime5)

    QVAS_Image = ET.Element('QVAS_Image')
    QVAS_Success = ET.SubElement(QVAS_Image, 'QVAS_Success')
    QVAS_Success.text = 'Success'
    QVAS_ProbPath = ET.SubElement(QVAS_Image, 'QVAS_ProbPath')
    QVAS_ProbPath.text = DOWNLOAD_LINK + 'seg_' + filename
    QVAS_SegPath = ET.SubElement(QVAS_Image, 'QVAS_SegPath')
    QVAS_SegPath.text = DOWNLOAD_LINK + 'seg_' + filename
    if 0:  # probimg:
        QVAS_ProbImg = ET.SubElement(QVAS_Image, 'QVAS_ProbImg')
        # QVAS_ProbImg.text = test_prob_str

    contourin, contourout = toctbd(polarbdrestore, ysize // 2 * ENLARGEMENT, xsize // 2 * ENLARGEMENT)
    contours = [contourin, contourout]
    for conti in range(len(contours)):
        QVAS_Contour = ET.SubElement(QVAS_Image, 'QVAS_Contour')
        Contour_Point = ET.SubElement(QVAS_Contour, 'Contour_Point')
        if conti == 1:
            ct = "OuterWall"
        else:
            ct = "Lumen"
        contourfilename = os.path.join(SAVE_FOLDER, ct + '.txt')
        contourfile = open(contourfilename, "w")
        for coutnodei in contours[conti]:
            contourfile.write("%.2f %.2f\n" % (coutnodei[0], coutnodei[1]))
        # repeat first point
        contourfile.write("%.2f %.2f\n" % (contours[conti][0][0], contours[conti][0][1]))
        contourfile.close()

        ContourType = ET.SubElement(QVAS_Contour, 'ContourType')
        ContourType.text = ct
        for coutnodei in contours[conti]:
            Point = ET.SubElement(Contour_Point, 'Point')
            Point.set('x', str((coutnodei[0] / ENLARGEMENT + xcenter - xsize // 2) / scaleres))
            Point.set('y', str((coutnodei[1] / ENLARGEMENT + ycenter - ysize // 2) / scaleres))

    endalltime = datetime.datetime.now()
    elaspsealltime6 = endalltime - starttime
    print('contours saved', elaspsealltime6)

    xml = ET.tostring(QVAS_Image)
    myfile = open(SAVE_FOLDER + "contour.xml", "w")
    myfile.write(xml.decode("utf-8"))
    myfile.close()
    return xml


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

    def toPolar(self, cart_img):
        # cart img in original size, height,width,depth
        polar_img = np.zeros((self.cfg['height'], self.cfg['width'], cart_img.shape[2]))
        for di in range(cart_img.shape[2]):
            cart_img_enlarge = cv2.resize(cart_img[:, :, di], (0, 0), fx=4 * self.cfg['SCALE'],
                                          fy=4 * self.cfg['SCALE'])
            polar_img[:, :, di] = topolar(cart_img_enlarge / np.max(cart_img_enlarge), self.cfg['width'],
                                          self.cfg['height'])
        return polar_img


import keras
from keras.layers import Input, Dense, Conv3D, Conv2D, Reshape, MaxPooling3D, MaxPooling2D, UpSampling3D, UpSampling2D, \
    BatchNormalization, Flatten, Dropout, concatenate
from keras.models import Model
from keras import backend as K


class PolarReg(PolarBase):
    def __init__(self, model_path=None):
        cfg = {}
        # ICA model
        # cfg['modelname'] = 'ICAReg1-8'
        # 3d merge
        cfg['modelname'] = 'UnetRGB15-3'

        if cfg['modelname'] == 'ICAReg1-8':
            cfg['modelpath'] = '//Desktop4/Dtensorflow/LiChen/VW/ICAReg/ICAReg1-8/Epo122-0.04013-0.77108.hdf5'
            cfg['height'] = 256
        elif cfg['modelname'] == 'UnetRGB15-3':
            cfg['modelpath'] = '//Desktop4/Dtensorflow/LiChen/VW/LatteReg/UnetRGB15-3/ModelEpo314-0.07959-0.77438.hdf5'
            cfg['height'] = 64

        cfg['width'] = 256
        cfg['patchheight'] = cfg['height']
        cfg['depth'] = 3
        cfg['channel'] = 1
        cfg['SCALE'] = 1
        cfg['vw_pred_type'] = 'reg'
        self.cfg = cfg
        # load model
        self.loadModel(model_path)
        self.resetImg()

    def setModel(self, model_path):
        self.cfg['modelname'] = model_path.replace('\\', '/')[-2]
        self.cfg['modelpath'] = model_path

    def loadModel(self, model_path=None):
        def polarloss(y_true, y_pred):
            dmax = K.sum(tf.math.maximum(y_true, y_pred))
            dmin = K.sum(tf.math.minimum(y_true, y_pred))
            return K.log(tf.math.divide(dmax, dmin))

        if model_path is not None:
            self.setModel(model_path)
        print('loading model from', self.cfg['modelpath'])
        self.model = self.buildmodel(self.cfg)
        self.model.load_weights(self.cfg['modelpath'])
        if self.model.input_shape != (
                None, self.cfg['height'], self.cfg['width'], self.cfg['depth'], self.cfg['channel']):
            self.cfg['height'], self.cfg['width'], self.cfg['depth'], self.cfg['channel'] = self.model.input_shape[1:]
            self.cfg['patchheight'] = self.cfg['height']
            print('set model shape to', self.model.input_shape)
        self.feat_model = None
        print('model shape', self.model.input_shape)

    def buildmodel(self, config):
        K.clear_session()
        # pred img and cont at the same time
        input_img = Input(shape=(config['height'], config['width'], config['depth'], config['channel']),
                          name='patchimg')
        # input_pos = Input(shape=(2,), name='origin')

        af = 'relu'
        kif = 'glorot_normal'

        '''conv3a_1 = Conv3D(32, (3, 3, 3), activation=af, kernel_initializer=kif, padding='same',name='con3a_1')(input_img)
        conv3a_2 = Conv3D(32, (3, 3, 3), activation=af, kernel_initializer=kif, padding='same',name='con3a_2')(conv3a_1)
        conv3a_2 = BatchNormalization()(conv3a_2)
        mp3a = MaxPooling3D((2, 2, 1), padding='same', name='pooling3a')(conv3a_2)'''

        conv1_1 = Conv3D(32, (3, 3, 3), activation=af, kernel_initializer=kif, padding='same', name='con1_1')(input_img)
        conv1_2 = Conv3D(32, (3, 3, 3), activation=af, kernel_initializer=kif, padding='same', name='con1_2')(conv1_1)
        conv1_2 = BatchNormalization()(conv1_2)
        mp1 = MaxPooling3D((2, 2, 1), padding='same', name='pooling1')(conv1_2)
        conv2_1 = Conv3D(64, (3, 3, 3), activation=af, kernel_initializer=kif, padding='same', name='con2_1')(mp1)
        conv2_2 = Conv3D(64, (3, 3, 3), activation=af, kernel_initializer=kif, padding='same', name='con2_2')(conv2_1)
        conv2_2 = BatchNormalization()(conv2_2)
        mp2 = MaxPooling3D((2, 2, 1), padding='same', name='pooling2')(conv2_2)
        conv3_1 = Conv3D(128, (3, 3, 3), activation=af, kernel_initializer=kif, padding='same', name='con3_1')(mp2)
        conv3_2 = Conv3D(128, (3, 3, 3), activation=af, kernel_initializer=kif, padding='same', name='con3_2')(conv3_1)
        conv3_2 = BatchNormalization()(conv3_2)
        mp3 = MaxPooling3D((2, 2, 1), padding='same', name='pooling3')(conv3_2)

        conv4_1 = Conv3D(256, (3, 3, 3), activation=af, kernel_initializer=kif, padding='same', name='con4_1')(mp3)

        convr1_1 = Conv3D(128, (3, 3, 3), activation=af, kernel_initializer=kif, padding='same', name='conr1_1')(
            conv4_1)
        # convr1_2 = Conv3D(128, (3, 3, 3), activation=af, kernel_initializer=kif, padding='same',name='conr1_2')(convr1_1)
        convr1_2 = BatchNormalization()(convr1_1)
        mp1 = MaxPooling3D((2, 2, 1), padding='same', name='poolingr1')(convr1_2)
        convr2_1 = Conv3D(64, (3, 3, 3), activation=af, kernel_initializer=kif, padding='same', name='conr2_1')(mp1)
        convr2_2 = Conv3D(64, (3, 3, 3), activation=af, kernel_initializer=kif, padding='same', name='conr2_2')(
            convr2_1)
        convr2_2 = BatchNormalization()(convr2_2)
        mp2 = MaxPooling3D((1, 2, 1), padding='same', name='poolingr2')(convr2_2)
        convr3_1 = Conv3D(32, (3, 3, 3), activation=af, kernel_initializer=kif, padding='same', name='conr3_1')(mp2)
        convr3_2 = Conv3D(32, (3, 3, 3), activation=af, kernel_initializer=kif, padding='same', name='conr3_2')(
            convr3_1)
        convr3_2 = BatchNormalization()(convr3_2)
        mp3 = MaxPooling3D((1, 2, 1), padding='same', name='poolingr3')(convr3_2)
        convr4_1 = Conv3D(16, (3, 3, 3), activation=af, kernel_initializer=kif, padding='same', name='conr4_1')(mp3)
        # convr4_2 = Conv3D(32, (3, 3, 3), activation=af, kernel_initializer=kif, padding='same',name='conr4_2')(convr4_1)
        regresser = Conv3D(16, (3, 3, 3), activation='sigmoid', padding='same', name='regresser')(convr4_1)
        print(regresser)

        fcn1 = Flatten(name='aux_fx1')(regresser)
        fcn1 = Dropout(0.2)(fcn1)
        if 'regnum' in config:
            regnum = config['regnum']
        else:
            regnum = 2
        regr = Dense(config['patchheight'] * regnum, name='aux_outputr')(fcn1)  # ,activation='softmax'
        regr = Reshape((config['patchheight'], regnum), name='reg')(regr)

        if 'gradinput' in config and config['gradinput']:
            input_grad = Input(shape=(config['height'], config['width']),
                               name='patchimg')
            cnn = Model(inputs=[input_grad, input_grad], outputs=regr)
        else:
            cnn = Model(inputs=input_img, outputs=regr)

        # lr=0.01, momentum=0.9,nesterov =True

        return cnn

    def predict(self, polarimg):
        # polarimg shape of
        if polarimg.shape != (self.cfg['height'], self.cfg['width'], self.cfg['depth']):
            raise ValueError('Image input size mismatch with model', polarimg.shape,
                             (self.cfg['height'], self.cfg['width'], self.cfg['depth']))
        self.resetImg()
        self.polarpatch = polarimg
        # add channel dimension
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

    # pred long height patches with sliding window
    def polar_pred_long_patch(self, polar_img):
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

    def featPred(self, polar_patch):
        if self.feat_model is None:
            self.feat_model = keras.Model(inputs=self.model.input,
                                          outputs=self.model.get_layer('aux_fx1').output)
        features = self.feat_model.predict(polar_patch[None, :, :, :, None])
        return features[0]

    def featPredStack(self, polar_patch):
        if self.feat_model is None:
            self.feat_model = keras.Model(inputs=self.model.input,
                                          outputs=self.model.get_layer('aux_fx1').output)
        polar_stack = np.zeros((polar_patch.shape[2], self.cfg['width'], self.cfg['height'], self.cfg['depth'], 1))
        for i in range(polar_patch.shape[2]):
            if i == 0:
                polar_stack[i, :, :, 0, 0] = polar_patch[:, :, 0]
                polar_stack[i, :, :, 1, 0] = polar_patch[:, :, 0]
                polar_stack[i, :, :, 2, 0] = polar_patch[:, :, 1]
            elif i == polar_patch.shape[2] - 1:
                polar_stack[i, :, :, 0, 0] = polar_patch[:, :, i - 1]
                polar_stack[i, :, :, 1, 0] = polar_patch[:, :, i]
                polar_stack[i, :, :, 2, 0] = polar_patch[:, :, i]
            else:
                polar_stack[i, :, :, :, 0] = polar_patch[:, :, i - 1:i + 2]
        features = self.feat_model.predict(polar_stack)
        return features
