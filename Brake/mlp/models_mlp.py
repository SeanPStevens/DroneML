import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def make_mlp(dim, regress=False):
    # define our MLP network
    model = tf.keras.Sequential()
    model.add(layers.Dense(8, input_dim=dim, activation="relu"))
    model.add(layers.Dense(4, activation="relu"))

    # check to see if the regression node should be added
    if regress:
        model.add(layers.Dense(1, activation="linear"))

    # return our model
    return model

