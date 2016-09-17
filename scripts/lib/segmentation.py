import keras.backend as K

def binaryCE(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_pred, y_true), axis=None, keepdims=False)

def dice_coeff(y_true, y_pred):
    return K.mean((2 * K.sum(y_true * K.round(y_pred)) + 1) / (K.sum(y_true) + K.sum(K.round(y_pred)) + 1),
                  axis=None, keepdims=False)

def dice_coeff_loss(y_true, y_pred):
    return -dice_coeff(y_true, y_pred)