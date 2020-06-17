# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 10:22:09 2020

@author: Admin
"""
#shabake
from keras import backend as K
from keras import layers, models, optimizers
from keras.layers import Layer
from keras.layers import Input, Conv2D, Activation, Dense, Dropout, Lambda, Reshape, Concatenate
from keras.layers import BatchNormalization, MaxPooling2D, Flatten, Conv1D, Deconvolution2D, Conv2DTranspose
from keras.callbacks import Callback, ModelCheckpoint, TensorBoard
from keras.utils import plot_model
from keras.layers.convolutional import UpSampling2D
import numpy as np
import tensorflow as tf
import os

from capslayers import Conv2DCaps, ConvCapsuleLayer3D, CapsuleLayer, CapsToScalars, Mask_CID, ConvertToCaps, FlattenCaps

def DeepCapsNetTwoPath(input_shape1,input_shape2, n_class, routings):
    # assemble encoder
    
    x33 = Input(shape=input_shape1)
    x15 = Input(shape=input_shape2)
    path1 = Conv2D(128, (5, 5), strides=(1, 1), activation='relu', padding="same")(x33)  # common conv layer
    path1=MaxPooling2D(pool_size=(2, 2))(path1)
    path1=Dropout(0.5)(path1)
    path1 = ConvertToCaps()(path1)
    path1 = Conv2DCaps(16, 8, kernel_size=(5, 5), strides=(2, 2), r_num=1, b_alphas=[1, 1, 1])(path1)
    path1 = Conv2DCaps(16, 8, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1])(path1)

    
    path2 = Conv2D(128, (5, 5), strides=(1, 1), activation='relu', padding="same")(x15)  # common conv layer
    path2 = ConvertToCaps()(path2)
    path2 = Conv2DCaps(16, 8, kernel_size=(5, 5), strides=(2, 2), r_num=1, b_alphas=[1, 1, 1])(path2)
    x = layers.Add()([path1, path2])

    
    
    l=x

    # l = Conv2D(128, (5, 5), strides=(1, 1), activation='relu', padding="same")(l)  # common conv layer
    # # l = BatchNormalization()(l)
    # l = ConvertToCaps()(l)

    # l = Conv2DCaps(32, 4, kernel_size=(3, 3), strides=(2, 2), r_num=1, b_alphas=[1, 1, 1])(l)
    # l_skip = Conv2DCaps(32, 4, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1])(l)
    # l = Conv2DCaps(32, 4, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1])(l)
    # l = Conv2DCaps(32, 4, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1])(l)
    # l = layers.Add()([l, l_skip])

    l = Conv2DCaps(16, 8, kernel_size=(5, 5), strides=(2, 2), r_num=1, b_alphas=[1, 1, 1])(l)
    l_skip = Conv2DCaps(16, 8, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1])(l)
    l = Conv2DCaps(16, 8, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1])(l)
    l = layers.Add()([l, l_skip])

    # l = Conv2DCaps(32, 8, kernel_size=(3, 3), strides=(2, 2), r_num=1, b_alphas=[1, 1, 1])(l)
    # l_skip = Conv2DCaps(32, 8, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1])(l)
    # l = Conv2DCaps(32, 8, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1])(l)
    # l = Conv2DCaps(32, 8, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1])(l)
    # l = layers.Add()([l, l_skip])
    # l1 = l

    # l = Conv2DCaps(32, 8, kernel_size=(3, 3), strides=(2, 2), r_num=1, b_alphas=[1, 1, 1])(l)
    # l_skip = ConvCapsuleLayer3D(kernel_size=3, num_capsule=32, num_atoms=8, strides=1, padding='same', routings=3)(l)
    # l = Conv2DCaps(32, 8, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1])(l)
    # l = Conv2DCaps(32, 8, kernel_size=(3, 3), strides=(1, 1), r_num=1, b_alphas=[1, 1, 1])(l)
    # l = layers.Add()([l, l_skip])
    # l2 = l

    la = FlattenCaps()(l)
    # la = FlattenCaps()(l2)
    # lb = FlattenCaps()(l1)
    # l = layers.Concatenate(axis=-2)([la, lb])

#     l = Dropout(0.4)(l)
    digits_caps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings, channels=0, name='digit_caps')(la)

    l = CapsToScalars(name='capsnet')(digits_caps)

    m_capsnet = models.Model(inputs=[x33,x15], outputs=l, name='capsnet_model')

    y = Input(shape=(n_class,))

    masked_by_y = Mask_CID()([digits_caps, y])  
    masked = Mask_CID()(digits_caps)

    # Decoder Network
    decoder = models.Sequential(name='decoder')
    decoder.add(Dense(input_dim=16, activation="relu", output_dim=8 *8 * 16))
    decoder.add(Reshape((8, 8, 16)))
    decoder.add(BatchNormalization(momentum=0.8))
    decoder.add(Deconvolution2D(64, 3, 3, subsample=(1, 1), border_mode='same'))
    decoder.add(Deconvolution2D(32, 3, 3, subsample=(2, 2), border_mode='same'))
    decoder.add(Deconvolution2D(16, 3, 3, subsample=(1, 1), border_mode='same'))
    decoder.add(Deconvolution2D(4, 3, 3, subsample=(2, 2), border_mode='valid'))
    decoder.add(Activation("relu"))
    decoder.add(Reshape(target_shape=(33, 33, 4), name='out_recon'))

    train_model= models.Model([x33,x15, y], [m_capsnet.output, decoder(masked_by_y)])
    eval_model = models.Model([x33,x15], [m_capsnet.output, decoder(masked)])
    train_model.summary()
    return train_model, eval_model



