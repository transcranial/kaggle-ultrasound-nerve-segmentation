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
from inception_v4 import *

DATADIR = os.path.expanduser('~/data/kaggle/ultrasound-nerve-segmentation')
PROCDIR = os.path.expanduser('~/ml/kaggle/ultrasound-nerve-segmentation/processed')
MODELDIR = os.path.expanduser('~/ml/kaggle/ultrasound-nerve-segmentation/models')

WEIGHTS_PRESENCE_FILEPATH = os.path.join(MODELDIR, 'model-07b-presence-weights.hdf5')
HISTORY_PRESENCE_FILEPATH = os.path.join(MODELDIR, 'model-07b-presence-training-history.json')

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

    nb_filters_reduction_factor = 8
    
    # in original inception-v4, these are 4, 7, 3, respectively
    num_A_blocks = 1
    num_B_blocks = 2
    num_C_blocks = 1
    
    images_presence_b = Input(shape=(img_rows_presence, img_cols_presence, img_channels))

    x_b = inception_v4_stem(images_presence_b, nb_filters_reduction_factor=nb_filters_reduction_factor)
    for i in range(num_A_blocks):
        x_b = inception_v4_A(x_b, nb_filters_reduction_factor=nb_filters_reduction_factor)
    x_b = inception_v4_reduction_A(x_b, nb_filters_reduction_factor=nb_filters_reduction_factor)
    for i in range(num_B_blocks):
        x_b = inception_v4_B(x_b, nb_filters_reduction_factor=nb_filters_reduction_factor)
    x_b = inception_v4_reduction_B(x_b, nb_filters_reduction_factor=nb_filters_reduction_factor)
    for i in range(num_C_blocks):
        x_b = inception_v4_C(x_b, nb_filters_reduction_factor=nb_filters_reduction_factor)

    x_b = AveragePooling2D(pool_size=(4, 6), strides=(1, 1), border_mode='valid', dim_ordering='tf')(x_b)
    x_b = Dropout(0.5)(x_b)
    x_b = Flatten()(x_b)
    presence_b = Dense(1, activation='sigmoid')(x_b)

    model_presence_b = Model(input=images_presence_b, output=presence_b)

    model_presence_b.summary()

    print('')
    print('model init time: {}'.format(time.time() - start_time))

    start_time = time.time()

    model_presence_b.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

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

    history = model_presence_b.fit(data_images_train, data_presence_train,
                                   batch_size=batch_size, nb_epoch=nb_epoch, verbose=2,
                                   validation_data=(data_images_val, data_presence_val), shuffle=True,
                                   callbacks=[checkpointer, early_stopping])

    with open(HISTORY_PRESENCE_FILEPATH, 'w') as f_out:
        json.dump(history.history, f_out)


