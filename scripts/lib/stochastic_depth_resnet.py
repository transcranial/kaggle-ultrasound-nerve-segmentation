import numpy as np
import tensorflow as tf
from keras.layers import Dense, Activation, Flatten, Lambda, BatchNormalization, Reshape
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, UpSampling2D
from keras.engine import merge, Input, Model
import keras.backend as K

def get_p_survival(block=0, nb_total_blocks=110, p_survival_end=0.5, mode='linear_decay'):
    """
    See eq. (4) in stochastic depth paper: http://arxiv.org/pdf/1603.09382v1.pdf
    """
    if mode == 'uniform':
        return p_survival_end
    elif mode == 'linear_decay':
        return 1 - ((block + 1) / nb_total_blocks) * (1 - p_survival_end)
    else:
        raise


def zero_pad_channels(x, pad=0):
    """
    Function for Lambda layer
    """
    pattern = [[0, 0], [0, 0], [0, 0], [pad - pad // 2, pad // 2]]
    return tf.pad(x, pattern)


def stochastic_survival(y, p_survival=1.0):
    # binomial random variable
    survival = K.random_binomial((1,), p=p_survival)
    # during testing phase:
    # - scale y (see eq. (6))
    # - p_survival effectively becomes 1 for all layers (no layer dropout)
    return K.in_test_phase(tf.constant(p_survival, dtype='float32') * y,
                           survival * y)


def stochastic_depth_residual_block(x, nb_filters=16, block=0, nb_total_blocks=110, subsample_factor=1, dilation=1):
    """
    Stochastic depth paper: http://arxiv.org/pdf/1603.09382v1.pdf

    Residual block consisting of:
    - Conv - BN - ReLU - Conv - BN
    - identity shortcut connection
    - merge Conv path with shortcut path

    Original paper (http://arxiv.org/pdf/1512.03385v1.pdf) then has ReLU,
    but we leave this out: see https://github.com/gcr/torch-residual-networks

    Additional variants explored in http://arxiv.org/pdf/1603.05027v1.pdf

    some code adapted from https://github.com/dblN/stochastic_depth_keras
    """

    prev_nb_channels = K.int_shape(x)[3]

    filter_dilation = (dilation, dilation)

    if subsample_factor > 1:
        subsample = (subsample_factor, subsample_factor)
        # shortcut: subsample + zero-pad channel dim
        shortcut = AveragePooling2D(pool_size=subsample, dim_ordering='tf')(x)
    else:
        subsample = (1, 1)
        # shortcut: identity
        shortcut = x

    if nb_filters > prev_nb_channels:
        shortcut = Lambda(zero_pad_channels,
                          arguments={'pad': nb_filters - prev_nb_channels})(shortcut)

    y = Convolution2D(nb_filters, 3, 3, subsample=subsample, filter_dilation=(1, 1),
                      init='he_normal', border_mode='same', dim_ordering='tf')(x)
    y = BatchNormalization(axis=3)(y)
    y = Activation('relu')(y)
    y = Convolution2D(nb_filters, 3, 3, subsample=(1, 1), filter_dilation=filter_dilation,
                      init='he_normal', border_mode='same', dim_ordering='tf')(y)
    y = BatchNormalization(axis=3)(y)

    p_survival = get_p_survival(block=block, nb_total_blocks=nb_total_blocks, p_survival_end=0.5, mode='linear_decay')
    y = Lambda(stochastic_survival, arguments={'p_survival': p_survival})(y)

    if nb_filters < prev_nb_channels:
        y = Lambda(zero_pad_channels, arguments={'pad': prev_nb_channels - nb_filters})(y)

    out = merge([y, shortcut], mode='sum')

    return out
