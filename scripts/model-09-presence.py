"""
Run: nohup python -u xxx.py >xxx.out.log 2>xxx.err.log &
"""

import os
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Activation, Flatten, Lambda, BatchNormalization, Reshape, Dropout
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, UpSampling2D
from keras.engine import merge, Input, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
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

WEIGHTS_PRESENCE_FILEPATH = os.path.join(MODELDIR, 'model-09-presence-weights.hdf5')
HISTORY_PRESENCE_FILEPATH = os.path.join(MODELDIR, 'model-09-presence-training-history.json')

#############################################################################
# LOAD DATA

with open(os.path.join(PROCDIR, 'data_train_val_09.pkl'), 'rb') as f:
    (data_images_train, data_masks_train, data_presence_train, 
     data_images_val, data_masks_val, data_presence_val) = pickle.load(f)

print('')
print('image shapes:', data_images_train.shape, data_images_val.shape)
print('mask shapes:', data_masks_train.shape, data_masks_val.shape)
print('presence label shapes:', data_presence_train.shape, data_presence_val.shape)
print('')


with tf.device('/gpu:0'):

    #############################################################################
    # MODEL SETUP

    start_time = time.time()

    orig_img_rows, orig_img_cols = 420, 580
    img_rows, img_cols = 128, 128
    img_channels = 1

    blocks_per_group = 3
    nb_total_blocks = 5 * blocks_per_group
    
    images_presence = Input(shape=(img_rows, img_cols, img_channels))

    x = Convolution2D(16, 3, 3, subsample=(1, 1),
                      init='he_normal', border_mode='same', dim_ordering='tf')(images_presence)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    for i in range(0, blocks_per_group):
        nb_filters = 16
        x = stochastic_depth_residual_block(x, nb_filters=nb_filters,
                                            block=i, nb_total_blocks=nb_total_blocks,
                                            subsample_factor=1)

    for i in range(0, blocks_per_group):
        nb_filters = 32
        if i == 0:
            subsample_factor = 2
        else:
            subsample_factor = 1
        x = stochastic_depth_residual_block(x, nb_filters=nb_filters,
                                            block=blocks_per_group + i, nb_total_blocks=nb_total_blocks,
                                            subsample_factor=subsample_factor)

    for i in range(0, blocks_per_group):
        nb_filters = 64
        if i == 0:
            subsample_factor = 2
        else:
            subsample_factor = 1
        x = stochastic_depth_residual_block(x, nb_filters=nb_filters,
                                            block=2 * blocks_per_group + i, nb_total_blocks=nb_total_blocks,
                                            subsample_factor=subsample_factor)

    for i in range(0, blocks_per_group):
        nb_filters = 128
        if i == 0:
            subsample_factor = 2
        else:
            subsample_factor = 1
        x = stochastic_depth_residual_block(x, nb_filters=nb_filters,
                                            block=3 * blocks_per_group + i, nb_total_blocks=nb_total_blocks,
                                            subsample_factor=subsample_factor)

    for i in range(0, blocks_per_group):
        nb_filters = 256
        if i == 0:
            subsample_factor = 2
        else:
            subsample_factor = 1
        x = stochastic_depth_residual_block(x, nb_filters=nb_filters,
                                            block=4 * blocks_per_group + i, nb_total_blocks=nb_total_blocks,
                                            subsample_factor=subsample_factor)

    x = AveragePooling2D(pool_size=(8, 8), strides=None, border_mode='valid', dim_ordering='tf')(x)
    x = Flatten()(x)
    presence = Dense(1, activation='sigmoid')(x)

    model_presence = Model(input=images_presence, output=presence)

    model_presence.summary()

    print('')
    print('model init time: {}'.format(time.time() - start_time))

    start_time = time.time()

    model_presence.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    print('model compile time: {}'.format(time.time() - start_time))
    print('')

    #############################################################################
    # TRAINING

    batch_size = 64
    nb_epoch = 100

    # Model saving callback
    checkpointer = ModelCheckpoint(filepath=WEIGHTS_PRESENCE_FILEPATH, verbose=1, save_best_only=True)

    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=7)

    history = model_presence.fit(data_images_train, data_presence_train,
                                 batch_size=batch_size, nb_epoch=nb_epoch, verbose=2,
                                 validation_data=(data_images_val, data_presence_val), shuffle=True,
                                 callbacks=[checkpointer, early_stopping])

    with open(HISTORY_PRESENCE_FILEPATH, 'w') as f_out:
        json.dump(history.history, f_out)


