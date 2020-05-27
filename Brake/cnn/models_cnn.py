import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def make_cnn(img_size, filters=(16, 32, 64), regress=False):
    inputShape = (img_size, img_size, 3)

    inputs = keras.Input(shape=inputShape)

    for (i, f) in enumerate(filters):
        if i == 0:
            x = inputs

        x = layers.Conv2D(f, (3, 3), padding="same")(x)
        x = layers.Activation("relu")(x)
        x = layers.BatchNormalization(axis=-1)(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(16)(x)
    x = layers.Activation("relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(4)(x)
    x = layers.Activation("relu")(x)

    if regress:
        x = layers.Dense(1, activation="linear")(x)

    model = keras.Model(inputs, x)

    return model
