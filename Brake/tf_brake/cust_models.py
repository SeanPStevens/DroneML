import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def test_mlp(dim):
    dimmy = dim
    inputs = keras.Input()
    x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
    x = layers.Dense(64, activation="relu", name="dense_2")(x)
    outputs = layers.Dense(10, name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


def my_mlp(dim):
    inputs = keras.Input(shape=dim)
    x = layers.Dense(8, activation="relu", name="dense_1")(inputs)
    x = layers.Dense(1, activation="relu", name="dense_2")(x)
    outputs = layers.Dense(1, activation="linear", name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model
