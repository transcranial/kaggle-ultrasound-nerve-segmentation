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

WEIGHTS_SEGMENT_FILEPATH = os.path.join(MODELDIR, 'model-09-segment-weights.hdf5')
HISTORY_SEGMENT_FILEPATH = os.path.join(MODELDIR, 'model-09-segment-training-history.json')

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


with tf.device('/gpu:1'):

    #############################################################################
    # MODEL SETUP

    start_time = time.time()

    orig_img_rows, orig_img_cols = 420, 580
    img_rows_segment, img_cols_segment = 128, 128
    img_channels = 1

    blocks_per_group = 3
    nb_total_blocks = 5 * blocks_per_group

    images_segment = Input(shape=(img_rows_segment, img_cols_segment, img_channels))

    x = Convolution2D(16, 3, 3, subsample=(1, 1),
                      init='he_normal', border_mode='same', dim_ordering='tf')(images_segment)
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

    f2 = x
    f2 = UpSampling2D(size=(2, 2), dim_ordering='tf')(f2)
    f2 = Convolution2D(16, 3, 3, init='he_uniform', border_mode='same', activation='relu', dim_ordering='tf')(f2)

    for i in range(0, blocks_per_group):
        nb_filters = 64
        if i == 0:
            subsample_factor = 2
        else:
            subsample_factor = 1
        x = stochastic_depth_residual_block(x, nb_filters=nb_filters,
                                            block=2 * blocks_per_group + i, nb_total_blocks=nb_total_blocks,
                                            subsample_factor=subsample_factor)

    f4 = x
    f4 = UpSampling2D(size=(4, 4), dim_ordering='tf')(f4)
    f4 = Convolution2D(16, 3, 3, init='he_uniform', border_mode='same', activation='relu', dim_ordering='tf')(f4)

    for i in range(0, blocks_per_group):
        nb_filters = 128
        if i == 0:
            subsample_factor = 2
        else:
            subsample_factor = 1
        x = stochastic_depth_residual_block(x, nb_filters=nb_filters,
                                            block=3 * blocks_per_group + i, nb_total_blocks=nb_total_blocks,
                                            subsample_factor=subsample_factor)

    f8 = x
    f8 = UpSampling2D(size=(8, 8), dim_ordering='tf')(f8)
    f8 = Convolution2D(16, 3, 3, init='he_uniform', border_mode='same', activation='relu', dim_ordering='tf')(f8)

    for i in range(0, blocks_per_group):
        nb_filters = 256
        if i == 0:
            subsample_factor = 2
        else:
            subsample_factor = 1
        x = stochastic_depth_residual_block(x, nb_filters=nb_filters,
                                            block=4 * blocks_per_group + i, nb_total_blocks=nb_total_blocks,
                                            subsample_factor=subsample_factor)

    f16 = x
    f16 = UpSampling2D(size=(16, 16), dim_ordering='tf')(f16)
    f16 = Convolution2D(16, 3, 3, init='he_uniform', border_mode='same', activation='relu', dim_ordering='tf')(f16)

    segmentation = merge([f16, f8, f4, f2], mode='concat', concat_axis=-1)
    segmentation = Convolution2D(1, 1, 1,
                                 activation='sigmoid',
                                 init='uniform', border_mode='same', dim_ordering='tf')(segmentation)
    segmentation = Reshape((img_rows_segment, img_cols_segment))(segmentation)

    model_segment = Model(input=images_segment, output=segmentation)

    model_segment.summary()

    print('')
    print('model init time: {}'.format(time.time() - start_time))

    start_time = time.time()

    model_segment.compile(optimizer='rmsprop', loss=binaryCE, metrics=[dice_coeff])

    print('model compile time: {}'.format(time.time() - start_time))
    print('')

    #############################################################################
    # TRAINING

    batch_size = 48
    nb_epoch = 100

    # Model saving callback
    checkpointer = ModelCheckpoint(filepath=WEIGHTS_SEGMENT_FILEPATH, verbose=1, save_best_only=True)

    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    history = model_segment.fit(data_images_train, data_masks_train,
                                 batch_size=batch_size, nb_epoch=nb_epoch, verbose=2, shuffle=True,
                                 validation_data=(data_images_val, data_masks_val),
                                 callbacks=[checkpointer, early_stopping])

    with open(HISTORY_SEGMENT_FILEPATH, 'w') as f_out:
        json.dump(history.history, f_out)


