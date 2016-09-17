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

WEIGHTS_FILEPATH = os.path.join(MODELDIR, 'model-02-weights.hdf5')
HISTORY_FILEPATH = os.path.join(MODELDIR, 'model-02-trainhist.json')

#############################################################################
# LOAD DATA

with open(os.path.join(PROCDIR, 'data_train_val_02.pkl'), 'rb') as f:
    data_images_train, data_masks_train, data_images_val, data_masks_val = pickle.load(f)

print('')
print('Train/Validation data')
print('image shapes:', data_images_train.shape, data_images_val.shape)
print('mask shapes:', data_masks_train.shape, data_masks_val.shape)
print('')


#############################################################################
# MODEL SETUP

start_time = time.time()

orig_img_rows, orig_img_cols = 420, 580
img_rows, img_cols = 128, 192
img_channels = 1

blocks_per_group = 8
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

    f2 = x
    f2 = UpSampling2D(size=(2, 2), dim_ordering='tf')(f2)
    f2 = Convolution2D(8, 3, 3, init='he_uniform', border_mode='same', activation='relu', dim_ordering='tf')(f2)

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
    f4 = UpSampling2D(size=(4, 4), dim_ordering='tf')(f4)
    f4 = Convolution2D(8, 3, 3, init='he_uniform', border_mode='same', activation='relu', dim_ordering='tf')(f4)
        
with tf.device('/gpu:1'):

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
    f8 = UpSampling2D(size=(8, 8), dim_ordering='tf')(f8)
    f8 = Convolution2D(8, 3, 3, init='he_uniform', border_mode='same', activation='relu', dim_ordering='tf')(f8)

    for i in range(0, blocks_per_group):
        nb_filters = 64
        if i == 0:
            subsample_factor = 2
        else:
            subsample_factor = 1
        x = stochastic_depth_residual_block(x, nb_filters=nb_filters,
                                            block=4 * blocks_per_group + i, nb_total_blocks=nb_total_blocks,
                                            subsample_factor=subsample_factor)

    f16 = x
    f16 = UpSampling2D(size=(16, 16), dim_ordering='tf')(f16)
    f16 = Convolution2D(8, 3, 3, init='he_uniform', border_mode='same', activation='relu', dim_ordering='tf')(f16)

    segmentation = merge([f16, f8, f4, f2], mode='concat', concat_axis=-1)    
    segmentation = Convolution2D(1, 1, 1, 
                                 activation='sigmoid', 
                                 init='uniform', border_mode='same', dim_ordering='tf')(segmentation)
    segmentation = Reshape((img_rows, img_cols))(segmentation)

    model = Model(input=images, output=segmentation)

model.summary()

print('')
print('model init time: {}'.format(time.time() - start_time))

start_time = time.time()

model.compile(optimizer='adam',
              loss=binaryCE,
              metrics=[dice_coeff])

print('model compile time: {}'.format(time.time() - start_time))
print('')

#############################################################################
# TRAINING

batch_size = 48
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

#rand_i = random.randint(0, data_images_val.shape[0] - 1)
#plt.figure(figsize=(15,5))
#plt.subplot(1,3,1)
#plt.imshow(data_images_val[rand_i, :, :, 0], cmap=plt.cm.viridis)
#plt.subplot(1,3,2)
#plt.title('mask pred')
#plt.imshow(model.predict(data_images_val[rand_i:rand_i+1], verbose=0)[0, :, :], cmap=plt.cm.bone)
#plt.subplot(1,3,3)
#plt.title('mask true')
#plt.imshow(data_masks_val[rand_i, :, :], cmap=plt.cm.bone)
#plt.savefig('sample.png', bbox_inches='tight')

