"""
Run: KERAS_BACKEND=theano THEANO_FLAGS='floatX=float32,device=gpu0,nvcc.fastmath=True,optimizer=fast_run' python -u xxx.py >xxx.out.log 2>xxx.err.log &
"""

import os
import numpy as np
from keras.layers import Dense, Activation, Flatten, Lambda, BatchNormalization, Reshape, Dropout, ELU
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, UpSampling2D
from keras.layers import LSTM, GRU, TimeDistributed
from keras.engine import merge, Input, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
import keras.backend as K
import time
import random
import pickle
import json

from helper import to_rle
from segmentation import *
from imagegen import CustomImageDataGenerator

# for preventing python max recursion limit error
import sys
sys.setrecursionlimit(50000)

DATADIR = os.path.expanduser('~/data/kaggle/ultrasound-nerve-segmentation')
PROCDIR = os.path.expanduser('~/ml/kaggle/ultrasound-nerve-segmentation/processed')
MODELDIR = os.path.expanduser('~/ml/kaggle/ultrasound-nerve-segmentation/models')

WEIGHTS_SEGMENT_FILEPATH = os.path.join(MODELDIR, 'model-11-segment-weights.hdf5')
HISTORY_SEGMENT_FILEPATH = os.path.join(MODELDIR, 'model-11-segment-training-history.json')

#############################################################################
# LOAD DATA

with open(os.path.join(PROCDIR, 'data_train_val_11.pkl'), 'rb') as f:
    (data_images_train, data_masks_train, data_presence_train, 
     data_images_val, data_masks_val, data_presence_val) = pickle.load(f)

print('')
print('image shapes:', data_images_train.shape, data_images_val.shape)
print('mask shapes:', data_masks_train.shape, data_masks_val.shape)
print('presence label shapes:', data_presence_train.shape, data_presence_val.shape)
print('')

#############################################################################
# MODEL SETUP

def rotate90(X, direction='clockwise'):
    if direction == 'clockwise':
        return X.transpose((0, 2, 1))[:, :, ::-1]
    elif direction == 'counterclockwise':
        return X.transpose((0, 2, 1))[:, ::-1, :]
    else:
        raise

start_time = time.time()

orig_img_rows, orig_img_cols = 420, 580
img_rows, img_cols = 256, 256
img_channels = 1

images = Input(shape=(img_channels, img_rows, img_cols))

x = Convolution2D(16, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th')(images)
x = BatchNormalization(axis=1)(x)
x = ELU()(x)
x = Convolution2D(16, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th')(x)
x = BatchNormalization(axis=1)(x)
x = ELU()(x)
x = MaxPooling2D(pool_size=(2,2), strides=None, border_mode='valid', dim_ordering='th')(x)

x = Convolution2D(32, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th')(x)
x = BatchNormalization(axis=1)(x)
x = ELU()(x)
x = Convolution2D(32, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th')(x)
x = BatchNormalization(axis=1)(x)
x = ELU()(x)
x = MaxPooling2D(pool_size=(2,2), strides=None, border_mode='valid', dim_ordering='th')(x)

x = Convolution2D(64, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th')(x)
x = BatchNormalization(axis=1)(x)
x = ELU()(x)
x = Convolution2D(64, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th')(x)
x = BatchNormalization(axis=1)(x)
x = ELU()(x)
x = Convolution2D(64, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th')(x)
x = BatchNormalization(axis=1)(x)
x = ELU()(x)
x = MaxPooling2D(pool_size=(2,2), strides=None, border_mode='valid', dim_ordering='th')(x)

x = Convolution2D(128, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th')(x)
x = BatchNormalization(axis=1)(x)
x = ELU()(x)
x = Convolution2D(128, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th')(x)
x = BatchNormalization(axis=1)(x)
x = ELU()(x)
x = Convolution2D(128, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th')(x)
x = BatchNormalization(axis=1)(x)
x = ELU()(x)
x = MaxPooling2D(pool_size=(2,2), strides=None, border_mode='valid', dim_ordering='th')(x)

x = Convolution2D(256, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th')(x)
x = BatchNormalization(axis=1)(x)
x = ELU()(x)
x = Convolution2D(256, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th')(x)
x = BatchNormalization(axis=1)(x)
x = ELU()(x)
x = Convolution2D(256, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th')(x)
x = BatchNormalization(axis=1)(x)
x = ELU()(x)
x = MaxPooling2D(pool_size=(2,2), strides=None, border_mode='valid', dim_ordering='th')(x)

x = Flatten()(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)
x = Reshape((64, 8, 8))(x)

x = UpSampling2D(size=(2, 2), dim_ordering='th')(x)
x = Convolution2D(256, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th')(x)
x = BatchNormalization(axis=1)(x)
x = ELU()(x)
x = Convolution2D(256, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th')(x)
x = BatchNormalization(axis=1)(x)
x = ELU()(x)
x = Convolution2D(256, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th')(x)
x = BatchNormalization(axis=1)(x)
x = ELU()(x)

x = UpSampling2D(size=(2, 2), dim_ordering='th')(x)
x = Convolution2D(128, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th')(x)
x = BatchNormalization(axis=1)(x)
x = ELU()(x)
x = Convolution2D(128, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th')(x)
x = BatchNormalization(axis=1)(x)
x = ELU()(x)
x = Convolution2D(128, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th')(x)
x = BatchNormalization(axis=1)(x)
x = ELU()(x)

x = UpSampling2D(size=(2, 2), dim_ordering='th')(x)
x = Convolution2D(64, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th')(x)
x = BatchNormalization(axis=1)(x)
x = ELU()(x)
x = Convolution2D(64, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th')(x)
x = BatchNormalization(axis=1)(x)
x = ELU()(x)
x = Convolution2D(64, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th')(x)
x = BatchNormalization(axis=1)(x)
x = ELU()(x)

x = UpSampling2D(size=(2, 2), dim_ordering='th')(x)
x = Convolution2D(32, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th')(x)
x = BatchNormalization(axis=1)(x)
x = ELU()(x)
x = Convolution2D(32, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th')(x)
x = BatchNormalization(axis=1)(x)
x = ELU()(x)

x = UpSampling2D(size=(2, 2), dim_ordering='th')(x)
x = Convolution2D(16, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th')(x)
x = BatchNormalization(axis=1)(x)
x = ELU()(x)
x = Convolution2D(16, 3, 3, init='he_uniform', border_mode='same', dim_ordering='th')(x)
x = BatchNormalization(axis=1)(x)
x = ELU()(x)

x = Convolution2D(1, 1, 1, activation='sigmoid', init='uniform', border_mode='same', dim_ordering='th')(x)
x = Reshape((img_rows, img_cols))(x)
x = Dropout(0.5)(x)

rnn_we = GRU(img_cols, activation='tanh', inner_activation='hard_sigmoid', return_sequences=True)(x)
rnn_ew = GRU(img_cols, activation='tanh', inner_activation='hard_sigmoid', go_backwards=True, return_sequences=True)(x)
rnn_we_ew = merge([rnn_we, rnn_ew], mode='concat', concat_axis=-1)
rnn_we_ew = TimeDistributed(Dense(img_cols, init='uniform', activation='sigmoid'))(rnn_we_ew)

x_rotate = Lambda(rotate90, arguments={'direction': 'counterclockwise'})(x)
rnn_ns = GRU(img_rows, activation='tanh', inner_activation='hard_sigmoid', return_sequences=True)(x_rotate)
rnn_sn = GRU(img_rows, activation='tanh', inner_activation='hard_sigmoid', go_backwards=True, return_sequences=True)(x_rotate)
rnn_ns_sn = merge([rnn_ns, rnn_sn], mode='concat', concat_axis=-1)
rnn_ns_sn = TimeDistributed(Dense(img_rows, init='uniform', activation='sigmoid'))(rnn_ns_sn)
rnn_ns_sn = Lambda(rotate90, arguments={'direction': 'clockwise'})(rnn_ns_sn)

segmentation = merge([rnn_we_ew, rnn_ns_sn], mode='mul')

model_segment = Model(input=images, output=segmentation)

model_segment.summary()

print('')
print('model init time: {}'.format(time.time() - start_time))

start_time = time.time()

model_segment.compile(optimizer='adam', loss=binaryCE, metrics=[dice_coeff])

print('model compile time: {}'.format(time.time() - start_time))
print('')

#############################################################################
# TRAINING

batch_size = 32
nb_epoch = 100

datagen = CustomImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=0,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0,
    zoom_range=0.1,
    channel_shift_range=0.,
    fill_mode='constant',
    cval=0.,
    horizontal_flip=False,
    vertical_flip=False,
    dim_ordering='th')
    
datagen.fit(data_images_train)

# Model saving callback
checkpointer = ModelCheckpoint(filepath=WEIGHTS_SEGMENT_FILEPATH, verbose=1, save_best_only=True)

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

history = model_segment.fit_generator(datagen.flow(data_images_train, data_masks_train, 
                                                   batch_size=batch_size),
                                      samples_per_epoch=data_images_train.shape[0],
                                      nb_epoch=nb_epoch, verbose=2,
                                      validation_data=(data_images_val, data_masks_val[:,0,:,:]),
                                      callbacks=[checkpointer, early_stopping])

with open(HISTORY_SEGMENT_FILEPATH, 'w') as f_out:
    json.dump(history.history, f_out)


