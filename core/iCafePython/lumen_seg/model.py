import keras
from keras.models import Model
from keras.layers import Input, Dense, Conv3D, RepeatVector, Reshape, MaxPooling3D, UpSampling3D, add, multiply, concatenate
from keras.utils import multi_gpu_model
import tensorflow as tf


def build_3DUnet(width, height, depth, G=1, activationfun='relu', kernelinitfun='glorot_normal'):
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
        cnn = Model(inputs=input_img, outputs=[prob, dist])
    else:
        print("[INFO] training with {} GPUs...".format(G))
        with tf.device('/cpu:0'):
            # initialize the model
            s_cnn = Model(inputs=input_img, outputs=[prob, dist])
        # make the model parallel
        cnn = multi_gpu_model(s_cnn, gpus=G)

    # lr=0.01, momentum=0.9,nesterov =True
    #cnn.compile(optimizer='Adam', loss='binary_crossentropy')
    return cnn
