import datetime
import os

import keras
import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.layers import concatenate
from keras.layers.core import Dense
from keras.models import Model
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from gitdata import load_anno, load_imgs
from models import make_cnn, make_mlp

## Data paths
data_dir = "/Users/Sean/Documents/VSC/Brake"
anno_path = "{}/annotations/North1009.txt".format(data_dir)

## Some parameters
bs = 8
ep = 30
lr = 1e-3
lr_d = 1e-3 / ep
img_size = 64

NAME = "MLP_1_{}".format(datetime.datetime.now())
logs = "Brake/myBrake/logs/{}".format(NAME)
tensorboard = TensorBoard(log_dir=logs)


## Load numerical data
mm = MinMaxScaler()
anno = load_anno(txt_dir=anno_path)
shape_anno = anno.shape
df = np.zeros(shape=(shape_anno[0], shape_anno[1] - 2), dtype="float")
df[:, 0] = anno[:, 1]  # brake
df[:, 1] = anno[:, 2]  # ws
df[:, 2] = anno[:, 3]  # thot
df[:, 3] = anno[:, 5]  # acc

## Organize and split
split = train_test_split(df)
(t_anno, v_anno) = split

maxBrake = t_anno[:, 0].max()
t_brake = t_anno[:, 0] / maxBrake
v_brake = v_anno[:, 0] / maxBrake
t_anno = mm.fit_transform(t_anno)
v_anno = mm.transform(v_anno)

print("[ALERT] Data Loaded")

# model
mlp = make_mlp(t_anno.shape[1], regress=True)

model = Model(inputs=mlp.input, outputs=mlp.output)

print("[ALERT] Model Created")

opt = Adam(learning_rate=lr, decay=lr_d)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

print("[ALERT] Training Started")

model.fit(
    t_anno,
    t_brake,
    validation_data=(v_anno, v_brake),
    epochs=ep,
    batch_size=bs,
    callbacks=[tensorboard],
)

print("[ALERT] Training Done. Starting Predictions.")
preds = model.predict([v_anno])

diff = preds.flatten() - v_brake
percentDiff = (diff / v_brake) * 100
absPercentDiff = np.abs(percentDiff)
print("Percent Diff= {}%".format(absPercentDiff))

mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

print("Mean= {}%".format(mean))
print("std= {}".format(std))
