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
from imagegen import CustomImageDataGenerator

DATADIR = os.path.expanduser('~/data/kaggle/ultrasound-nerve-segmentation')
PROCDIR = os.path.expanduser('~/ml/kaggle/ultrasound-nerve-segmentation/processed')
MODELDIR = os.path.expanduser('~/ml/kaggle/ultrasound-nerve-segmentation/models')

WEIGHTS_SEGMENT_FILEPATH = os.path.join(MODELDIR, 'model-12-segment-weights.hdf5')
HISTORY_SEGMENT_FILEPATH = os.path.join(MODELDIR, 'model-12-segment-training-history.json')

#############################################################################
# LOAD DATA

with open(os.path.join(PROCDIR, 'data_train_val_12.pkl'), 'rb') as f:
    (data_patches_train, data_masks_train,
     data_patches_val, data_masks_val) = pickle.load(f)

print('')
print('image shapes:', data_patches_train.shape, data_patches_val.shape)
print('mask shapes:', data_masks_train.shape, data_masks_val.shape)
print('')
data_masks_train = np.expand_dims(data_masks_train, axis=3)
data_masks_val = np.expand_dims(data_masks_val, axis=3)

#############################################################################
# MODEL SETUP

start_time = time.time()

orig_img_rows, orig_img_cols = 420, 580
resized_img_rows, resized_img_cols = 128, 192
patch_rows, patch_cols = 64, 64
img_channels = 1

blocks_per_group = 3
nb_total_blocks = 9 * blocks_per_group

with tf.device('/gpu:0'):

    image_patch = Input(shape=(patch_rows, patch_cols, img_channels))

    x = Convolution2D(8, 3, 3, subsample=(1, 1),
                      init='he_normal', border_mode='same', dim_ordering='tf')(image_patch)
    x = BatchNormalization(axis=3)(x)
    x = ELU()(x)

    for i in range(0, blocks_per_group):
        nb_filters = 8
        x = stochastic_depth_residual_block(x, nb_filters=nb_filters,
                                            block=i, nb_total_blocks=nb_total_blocks,
                                            subsample_factor=1)
    f1 = x

    for i in range(0, blocks_per_group):
        nb_filters = 16
        if i == 0:
            subsample_factor = 2
        else:
            subsample_factor = 1
        x = stochastic_depth_residual_block(x, nb_filters=nb_filters,
                                            block=blocks_per_group + i, nb_total_blocks=nb_total_blocks,
                                            subsample_factor=subsample_factor)
    f2 = x

    for i in range(0, blocks_per_group):
        nb_filters = 32
        if i == 0:
            subsample_factor = 2
        else:
            subsample_factor = 1
        x = stochastic_depth_residual_block(x, nb_filters=nb_filters,
                                            block=2 * blocks_per_group + i, nb_total_blocks=nb_total_blocks,
                                            subsample_factor=subsample_factor)
    f4 = x

    for i in range(0, blocks_per_group):
        nb_filters = 64
        if i == 0:
            subsample_factor = 2
        else:
            subsample_factor = 1
        x = stochastic_depth_residual_block(x, nb_filters=nb_filters,
                                            block=3 * blocks_per_group + i, nb_total_blocks=nb_total_blocks,
                                            subsample_factor=subsample_factor)
    f8 = x

    for i in range(0, blocks_per_group):
        nb_filters = 128
        if i == 0:
            subsample_factor = 2
        else:
            subsample_factor = 1
        x = stochastic_depth_residual_block(x, nb_filters=nb_filters,
                                            block=4 * blocks_per_group + i, nb_total_blocks=nb_total_blocks,
                                            subsample_factor=subsample_factor)

with tf.device('/gpu:1'):

    x = UpSampling2D(size=(2, 2), dim_ordering='tf')(x)
    x = merge([x, f8], mode='concat', concat_axis=-1)
    x = Convolution2D(64, 3, 3, init='he_uniform', border_mode='same', dim_ordering='tf')(x)
    x = BatchNormalization(axis=3)(x)
    x = ELU()(x)
    for i in range(0, blocks_per_group):
        nb_filters = 64
        x = stochastic_depth_residual_block(x, nb_filters=nb_filters,
                                            block=5 * blocks_per_group + i, nb_total_blocks=nb_total_blocks,
                                            subsample_factor=1)

    x = UpSampling2D(size=(2, 2), dim_ordering='tf')(x)
    x = merge([x, f4], mode='concat', concat_axis=-1)
    x = Convolution2D(32, 3, 3, init='he_uniform', border_mode='same', dim_ordering='tf')(x)
    x = BatchNormalization(axis=3)(x)
    x = ELU()(x)
    for i in range(0, blocks_per_group):
        nb_filters = 32
        x = stochastic_depth_residual_block(x, nb_filters=nb_filters,
                                            block=6 * blocks_per_group + i, nb_total_blocks=nb_total_blocks,
                                            subsample_factor=1)

    x = UpSampling2D(size=(2, 2), dim_ordering='tf')(x)
    x = merge([x, f2], mode='concat', concat_axis=-1)
    x = Convolution2D(16, 3, 3, init='he_uniform', border_mode='same', dim_ordering='tf')(x)
    x = BatchNormalization(axis=3)(x)
    x = ELU()(x)
    for i in range(0, blocks_per_group):
        nb_filters = 16
        x = stochastic_depth_residual_block(x, nb_filters=nb_filters,
                                            block=7 * blocks_per_group + i, nb_total_blocks=nb_total_blocks,
                                            subsample_factor=1)

    x = UpSampling2D(size=(2, 2), dim_ordering='tf')(x)
    x = merge([x, f1], mode='concat', concat_axis=-1)
    x = Convolution2D(8, 3, 3, init='he_uniform', border_mode='same', dim_ordering='tf')(x)
    x = BatchNormalization(axis=3)(x)
    x = ELU()(x)
    for i in range(0, blocks_per_group):
        nb_filters = 8
        x = stochastic_depth_residual_block(x, nb_filters=nb_filters,
                                            block=8 * blocks_per_group + i, nb_total_blocks=nb_total_blocks,
                                            subsample_factor=1)

    segmentation = Convolution2D(1, 1, 1,
                                 activation='sigmoid',
                                 init='uniform', border_mode='same', dim_ordering='tf')(x)
    segmentation = Reshape((patch_rows, patch_cols))(segmentation)

model_segment = Model(input=image_patch, output=segmentation)

model_segment.summary()

print('')
print('model init time: {}'.format(time.time() - start_time))

start_time = time.time()

model_segment.compile(optimizer='adam', loss=binaryCE, metrics=[dice_coeff])

print('model compile time: {}'.format(time.time() - start_time))
print('')

#############################################################################
# TRAINING

batch_size = 128
nb_epoch = 200

datagen = CustomImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=0.,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.,
    zoom_range=0.,
    channel_shift_range=0.,
    fill_mode='constant',
    cval=0.,
    horizontal_flip=False,
    vertical_flip=False,
    dim_ordering='tf')
    
datagen.fit(data_patches_train)

# Model saving callback
checkpointer = ModelCheckpoint(filepath=WEIGHTS_SEGMENT_FILEPATH, verbose=1, save_best_only=True)

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

history = model_segment.fit_generator(datagen.flow(data_patches_train, data_masks_train, 
                                                   batch_size=batch_size),
                                      samples_per_epoch=data_patches_train.shape[0],
                                      nb_epoch=nb_epoch, verbose=2,
                                      validation_data=(data_patches_val, data_masks_val[:,:,:,0]),
                                      callbacks=[checkpointer, early_stopping])

with open(HISTORY_SEGMENT_FILEPATH, 'w') as f_out:
    json.dump(history.history, f_out)


