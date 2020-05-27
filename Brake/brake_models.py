import os

import absl
import keras
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.layers import *
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential


def brake_mlp(dim, regress=False):
    model = Sequential()
    model.add(Dense(8, input_dim=dim, activation="relu"))
    model.add(Dense(4, activation="relu"))

    if regress:
        model.add(Dense(1, activation="linear"))

    return model


def brake_cnn(img_size, filters=(16, 32, 64), regress=False):

    inputs = Input(shape=img_size)

    for (i, f) in enumerate(filters):
        if i == 0:
            x = inputs

        x = Conv2D(f, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(16)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(0.5)(x)

    x = Dense(4)(x)
    x = Activation("relu")(x)

    if regress:
        x = Dense(1, activation="linear")(x)

    model = Model(inputs, x)

    return model
