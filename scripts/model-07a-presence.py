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

WEIGHTS_PRESENCE_FILEPATH = os.path.join(MODELDIR, 'model-07a-presence-weights.hdf5')
HISTORY_PRESENCE_FILEPATH = os.path.join(MODELDIR, 'model-07a-presence-training-history.json')

#############################################################################
# LOAD DATA

with open(os.path.join(PROCDIR, 'data_train_val_07.pkl'), 'rb') as f:
    (data_images_train, data_images_val, 
     data_presence_train, data_presence_val, 
     data_masks_train, data_masks_val) = pickle.load(f)

print('')
print('image shapes:', data_images_train.shape, data_images_val.shape)
print('presence label shapes:', data_presence_train.shape, data_presence_val.shape)
print('mask shapes:', data_masks_train.shape, data_masks_val.shape)
print('')


with tf.device('/gpu:0'):

    #############################################################################
    # MODEL SETUP

    start_time = time.time()

    orig_img_rows, orig_img_cols = 420, 580
    img_rows_presence, img_cols_presence = 64, 96
    img_channels = 1

    blocks_per_group = 3
    nb_total_blocks = 5 * blocks_per_group
    
    images_presence_a = Input(shape=(img_rows_presence, img_cols_presence, img_channels))

    x_a = Convolution2D(4, 3, 3, subsample=(1, 1),
                      init='he_normal', border_mode='same', dim_ordering='tf')(images_presence_a)
    x_a = BatchNormalization(axis=3)(x_a)
    x_a = Activation('relu')(x_a)

    for i in range(0, blocks_per_group):
        nb_filters = 4
        x_a = stochastic_depth_residual_block(x_a, nb_filters=nb_filters,
                                            block=i, nb_total_blocks=nb_total_blocks,
                                            subsample_factor=1)

    for i in range(0, blocks_per_group):
        nb_filters = 8
        if i == 0:
            subsample_factor = 2
        else:
            subsample_factor = 1
        x_a = stochastic_depth_residual_block(x_a, nb_filters=nb_filters,
                                            block=blocks_per_group + i, nb_total_blocks=nb_total_blocks,
                                            subsample_factor=subsample_factor)

    for i in range(0, blocks_per_group):
        nb_filters = 16
        if i == 0:
            subsample_factor = 2
        else:
            subsample_factor = 1
        x_a = stochastic_depth_residual_block(x_a, nb_filters=nb_filters,
                                            block=2 * blocks_per_group + i, nb_total_blocks=nb_total_blocks,
                                            subsample_factor=subsample_factor)

    for i in range(0, blocks_per_group):
        nb_filters = 32
        if i == 0:
            subsample_factor = 2
        else:
            subsample_factor = 1
        x_a = stochastic_depth_residual_block(x_a, nb_filters=nb_filters,
                                            block=3 * blocks_per_group + i, nb_total_blocks=nb_total_blocks,
                                            subsample_factor=subsample_factor)

    for i in range(0, blocks_per_group):
        nb_filters = 64
        if i == 0:
            subsample_factor = 2
        else:
            subsample_factor = 1
        x_a = stochastic_depth_residual_block(x_a, nb_filters=nb_filters,
                                            block=4 * blocks_per_group + i, nb_total_blocks=nb_total_blocks,
                                            subsample_factor=subsample_factor)

    x_a = AveragePooling2D(pool_size=(4, 6), strides=None, border_mode='valid', dim_ordering='tf')(x_a)
    x_a = Dropout(0.5)(x_a)
    x_a = Flatten()(x_a)
    presence_a = Dense(1, activation='sigmoid')(x_a)

    model_presence_a = Model(input=images_presence_a, output=presence_a)

    model_presence_a.summary()

    print('')
    print('model init time: {}'.format(time.time() - start_time))

    start_time = time.time()

    model_presence_a.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print('model compile time: {}'.format(time.time() - start_time))
    print('')

    #############################################################################
    # TRAINING

    batch_size = 256
    nb_epoch = 50

    # Model saving callback
    checkpointer = ModelCheckpoint(filepath=WEIGHTS_PRESENCE_FILEPATH, verbose=1, save_best_only=True)

    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    history = model_presence_a.fit(data_images_train, data_presence_train,
                                   batch_size=batch_size, nb_epoch=nb_epoch, verbose=2,
                                   validation_data=(data_images_val, data_presence_val), shuffle=True,
                                   callbacks=[checkpointer, early_stopping])

    with open(HISTORY_PRESENCE_FILEPATH, 'w') as f_out:
        json.dump(history.history, f_out)


