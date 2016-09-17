"""
Run: nohup python -u xxx.py >xxx.out.log 2>xxx.err.log &
"""

import os

# allows matplotlib to directly save to file
import matplotlib
matplotlib.use('Agg')

import numpy as np
from skimage import io
from matplotlib import pyplot as plt
from scipy.misc import imresize
from scipy import ndimage
import tensorflow as tf
from keras.layers import Dense, Activation, Flatten, Lambda, BatchNormalization, Reshape
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, UpSampling2D
from keras.engine import merge, Input, Model
from keras.optimizers import SGD
from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import keras.backend as K
import time
import random
import pickle
import json

from helper import to_rle
from segmentation import *
from stochastic_depth_resnet import *

DATADIR = os.path.expanduser('~/data/kaggle/ultrasound-nerve-segmentation')
PROCDIR = os.path.expanduser('~/ml/kaggle/ultrasound-nerve-segmentation/processed')
MODELDIR = os.path.expanduser('~/ml/kaggle/ultrasound-nerve-segmentation/models')

VAE_WEIGHTS_FILEPATH = os.path.join(MODELDIR, 'model-vae-16-weights.hdf5')

WEIGHTS_FILEPATH = os.path.join(MODELDIR, 'model-16a-weights.hdf5')
HISTORY_FILEPATH = os.path.join(MODELDIR, 'model-16a-history.json')

#############################################################################
# LOAD DATA

with open(os.path.join(PROCDIR, 'data_train_val_16a.pkl'), 'rb') as f:
    data_images_train, data_masks_train, data_images_val, data_masks_val = pickle.load(f)

print('')
print('Train/Validation data')
print('image shapes:', data_images_train.shape, data_images_val.shape)
print('mask shapes:', data_masks_train.shape, data_masks_val.shape)
print('')


#############################################################################
# MODEL SETUP

decoder_layers = [
    Dense(12 * 16, activation='relu'),
    Reshape((12, 16, 1)),
    UpSampling2D(size=(2, 2), dim_ordering='tf'),
    Convolution2D(32, 3, 3, subsample=(1, 1), init='he_normal', activation='relu', 
                  border_mode='same', dim_ordering='tf'),
    Convolution2D(32, 3, 3, subsample=(1, 1), init='he_normal', activation='relu', 
                  border_mode='same', dim_ordering='tf'),
    UpSampling2D(size=(2, 2), dim_ordering='tf'),
    Convolution2D(16, 3, 3, subsample=(1, 1), init='he_normal', activation='relu', 
                  border_mode='same', dim_ordering='tf'),
    Convolution2D(16, 3, 3, subsample=(1, 1), init='he_normal', activation='relu', 
                  border_mode='same', dim_ordering='tf'),
    UpSampling2D(size=(2, 2), dim_ordering='tf'),
    Convolution2D(8, 3, 3, subsample=(1, 1), init='he_normal', activation='relu', 
                  border_mode='same', dim_ordering='tf'),
    Convolution2D(8, 3, 3, subsample=(1, 1), init='he_normal', activation='relu', 
                  border_mode='same', dim_ordering='tf'),
    Convolution2D(1, 1, 1, activation='sigmoid', 
                  init='uniform', border_mode='same', dim_ordering='tf'),
    Reshape((96, 128))
]

latent_dim = 10
epsilon_std = 0.001

masks = Input(batch_shape=(91, 96, 128))
x_encoder = Reshape((96, 128, 1), input_shape=(96, 128))(masks)
x_encoder = Convolution2D(8, 3, 3, subsample=(1, 1), init='he_normal', activation='relu', 
                          border_mode='same', dim_ordering='tf')(x_encoder)
x_encoder = Convolution2D(8, 3, 3, subsample=(1, 1), init='he_normal', activation='relu', 
                          border_mode='same', dim_ordering='tf')(x_encoder)
x_encoder = MaxPooling2D(pool_size=(2, 2), border_mode='valid', dim_ordering='tf')(x_encoder)
x_encoder = Convolution2D(16, 3, 3, subsample=(1, 1), init='he_normal', activation='relu', 
                          border_mode='same', dim_ordering='tf')(x_encoder)
x_encoder = Convolution2D(16, 3, 3, subsample=(1, 1), init='he_normal', activation='relu', 
                          border_mode='same', dim_ordering='tf')(x_encoder)
x_encoder = MaxPooling2D(pool_size=(2, 2), border_mode='valid', dim_ordering='tf')(x_encoder)
x_encoder = Convolution2D(32, 3, 3, subsample=(1, 1), init='he_normal', activation='relu', 
                          border_mode='same', dim_ordering='tf')(x_encoder)
x_encoder = Convolution2D(32, 3, 3, subsample=(1, 1), init='he_normal', activation='relu', 
                          border_mode='same', dim_ordering='tf')(x_encoder)
x_encoder = MaxPooling2D(pool_size=(2, 2), border_mode='valid', dim_ordering='tf')(x_encoder)
x_encoder = Flatten()(x_encoder)

z_mean = Dense(latent_dim)(x_encoder)
z_log_std = Dense(latent_dim)(x_encoder)

def sampling(args):
    z_mean, z_log_std = args
    epsilon = K.random_normal(shape=(91, latent_dim),
                              mean=0., std=epsilon_std)
    return z_mean + K.exp(z_log_std) * epsilon

z = Lambda(sampling)([z_mean, z_log_std])

x_decoder = z
for layer in decoder_layers:
    x_decoder = layer(x_decoder)
x_decoded_mean = x_decoder

def vae_loss(masks, x_decoded_mean):
    xent_loss = objectives.binary_crossentropy(masks, x_decoded_mean)
    kl_loss = - 0.5 * K.mean(1 + z_log_std - K.square(z_mean) - K.exp(z_log_std), axis=None, keepdims=False)
    return xent_loss + kl_loss

vae = Model(masks, x_decoded_mean)

vae.load_weights(VAE_WEIGHTS_FILEPATH)

start_time = time.time()

orig_img_rows, orig_img_cols = 420, 580
img_rows, img_cols = 96, 128
img_channels = 1

blocks_per_group = 4
nb_total_blocks = 5 * blocks_per_group

with tf.device('/gpu:0'):
    
    images = Input(shape=(img_rows, img_cols, img_channels))

    x = Convolution2D(8, 3, 3, subsample=(1, 1),
                      init='he_normal', border_mode='same', dim_ordering='tf')(images)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    for i in range(0, blocks_per_group):
        nb_filters = 8
        x = stochastic_depth_residual_block(x, nb_filters=nb_filters,
                                            block=i, nb_total_blocks=nb_total_blocks,
                                            subsample_factor=1)

    for i in range(0, blocks_per_group):
        nb_filters = 16
        if i == 0:
            subsample_factor = 2
        else:
            subsample_factor = 1
        x = stochastic_depth_residual_block(x, nb_filters=nb_filters,
                                            block=blocks_per_group + i, nb_total_blocks=nb_total_blocks,
                                            subsample_factor=subsample_factor)

    for i in range(0, blocks_per_group):
        nb_filters = 32
        if i == 0:
            subsample_factor = 2
        else:
            subsample_factor = 1
        x = stochastic_depth_residual_block(x, nb_filters=nb_filters,
                                            block=2 * blocks_per_group + i, nb_total_blocks=nb_total_blocks,
                                            subsample_factor=subsample_factor)
        
    for i in range(0, blocks_per_group):
        nb_filters = 64
        if i == 0:
            subsample_factor = 2
        else:
            subsample_factor = 1
        x = stochastic_depth_residual_block(x, nb_filters=nb_filters,
                                            block=3 * blocks_per_group + i, nb_total_blocks=nb_total_blocks,
                                            subsample_factor=subsample_factor)

with tf.device('/gpu:1'):
    
    for i in range(0, blocks_per_group):
        nb_filters = 64
        if i == 0:
            subsample_factor = 2
        else:
            subsample_factor = 1
        x = stochastic_depth_residual_block(x, nb_filters=nb_filters,
                                            block=4 * blocks_per_group + i, nb_total_blocks=nb_total_blocks,
                                            subsample_factor=subsample_factor)

    x = AveragePooling2D(pool_size=(6, 8), border_mode='valid', dim_ordering='tf')(x)
    x = Flatten()(x)
    z = Dense(latent_dim)(x)
    #presence = Dense(1, activation='sigmoid')(x)

    x_decoder = z
    for layer in decoder_layers:
        layer.trainable = False
        x_decoder = layer(x_decoder)
    segmentation = x_decoder

    #def presence_mul(args):
    #    p, s = args
    #    return K.expand_dims(p, dim=-1) * s
    #segmentation = Lambda(presence_mul)([presence, segmentation])

model = Model(input=images, output=segmentation)

print('model init time: {}'.format(time.time() - start_time))

start_time = time.time()

model.compile(optimizer='adam', loss=binaryCE, metrics=[dice_coeff])

print('model compile time: {}'.format(time.time() - start_time))


#############################################################################
# TRAINING

batch_size = 64
nb_epoch = 100

# Model saving callback
checkpointer = ModelCheckpoint(filepath=WEIGHTS_FILEPATH, verbose=1, save_best_only=True)

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(data_images_train, data_masks_train,
                    batch_size=batch_size, nb_epoch=nb_epoch, verbose=2,
                    validation_data=(data_images_val, data_masks_val), shuffle=True,
                    callbacks=[checkpointer, early_stopping])

with open(HISTORY_FILEPATH, 'w') as f_out:
    json.dump(history.history, f_out)