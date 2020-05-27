import os

import absl
import cv2
import keras
import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.layers import concatenate
from keras.layers.core import Dense
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
import mixed_regression.BrakeAI.brake_models as md

## dirs
data_dir = (
    "C:\\Users\\Sean\\Documents\\VSC\\mixed_regression\\BrakeAI\\Data\\Test42_Data"
)
img_dir = "{}\\imgs\\".format(data_dir)
anno_dir = "{}\\annotations\\North45.txt".format(data_dir)

oofs = np.loadtxt(anno_dir, dtype="float", delimiter=",")
img_names = []

# img names (without .jpg n shit) as int
for oof in oofs:
    img_names.append(int(oof[0]))


## organize numerical data for mlp
brake = oofs[:, 1]
wheel_speed = oofs[:, 2]
thot = oofs[:, 3]
SA = oofs[:, 4]
acc = oof[:, 5]

norm_acc = acc / acc.max()
norm_ws = wheel_speed / wheel_speed.max()
norm_thot = thot / thot.max()

norm_b = brake / brake.max()


## organize image data for cnn
# img_path = "{}\\{}.jpg".format(img_dir, int(oof[0]))

## Create mlp
mlp = md.brake_mlp(trainAnno.shape[1], regress=False)

## Create cnn
cnn = md.brake_cnn(64, 64, 3, regress=False)

## Concatenate networks
catInput = concatenate([mlp.output, cnn.output])

x = Dense(4, activation="relu")(catInput)
x = Dense(1, activation="linear")(x)

model = Model(inputs=[mlp.input, cnn.input], outputs=x)


## Train Model
op = keras.optimizers.Adam(learning_rate=1e-2, decay=1e-2 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer=op)

model.fit(
    [trainAnno, trainImgs],
    train_B,
    validation_data([valAnno, valImgs], val_B),
    epochs=200,
    batch_size=32,
)

## Some test predictions
