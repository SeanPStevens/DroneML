import os
import tensorflow as tf

import keras

# from tensorflow import keras
import absl
from keras.callbacks import TensorBoard
from keras.layers import Dense, Input, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential


def make_mlp2(dim, regress=False):

    # init motion input to functional model
    inputs = Input(shape=dim, name="motion_data")

    x = inputs
    x = Dense(16)(x)
    x = Activation("relu")(x)

    x = Dense(1)(x)
    x = Activation("relu")(x)

    if regress:
        x = Dense(1, activation="linear")(x)

    model = Model(inputs, x)

    return model


# def test_mlp(input_shape):  # shape=(784,)
#     inputs = keras.Input(shape=(784,), name="digits")
#     x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
#     x = layers.Dense(64, activation="relu", name="dense_2")(x)
#     outputs = layers.Dense(10, name="predictions")(x)
#     model = keras.Model(inputs=inputs, outputs=outputs)

#     return model


def make_mlp(dim, regress=False):
    model = Sequential()
    model.add(Dense(8, input_dim=dim, activation="relu"))
    model.add(Dense(1, activation="relu"))

    if regress:
        model.add(Dense(1, activation="linear"))

    return model


def make_cnn(img_size, filters=(16, 32, 64), regress=False):
    inputShape = (img_size, img_size, 3)

    inputs = Input(shape=inputShape)

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
