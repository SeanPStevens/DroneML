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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from gitdata import load_anno, load_imgs
from mixed_models import make_cnn, make_mlp


## Data paths
# data_dir = (
#     "C:\\Users\\Sean\\Documents\\VSC\\mixed_regression\\BrakeAI\\Data\\Test42_Data"
# )
# img_fold = "{}\\imgs".format(data_dir)
# anno_path = "{}\\annotations\\North45.txt".format(data_dir)
data_dir = "C:\\Users\\Sean\\Documents\\VSC\\myBrake\\Data\\NnS100_3"
img_fold = "{}\\imgs".format(data_dir)
anno_path = "{}\\annotations\\NS100_3.txt".format(data_dir)

## Some parameters
bs = 256
ep = 10
lr = 1e-2
lr_d = lr / ep
img_size = 64

NAME = "BrakeAI_100_{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
logs = "myBrake\\logs\\{}".format(NAME)
tensorboard = TensorBoard(log_dir=logs)


## Load numerical data
mm = MinMaxScaler()
anno = load_anno(txt_dir=anno_path)
shape_anno = anno.shape
# df = np.zeros(shape=(shape_anno[0], shape_anno[1] - 2), dtype="float")
df = np.zeros(shape=(shape_anno[0], shape_anno[1] - 1), dtype="float")
# df[:, 0] = anno[:, 1]  # brake
# df[:, 1] = anno[:, 2]  # ws
# df[:, 2] = anno[:, 3]  # thot
# df[:, 3] = anno[:, 5]  # acc

df[:, 0] = anno[:, 1]  # brake
df[:, 1] = anno[:, 2]  # ws
df[:, 2] = anno[:, 3]  # thot
df[:, 3] = anno[:, 4]  # acc


## Load Image data
imgs = load_imgs(annos=anno, imgs_dir=img_fold, target_size=(img_size, img_size))
imgs = imgs / 255.0

## Organize and split
split = train_test_split(df, imgs)
(t_anno, v_anno, t_img, v_img) = split

maxBrake = t_anno[:, 0].max()
t_brake = t_anno[:, 0] / maxBrake
v_brake = v_anno[:, 0] / maxBrake
t_anno = mm.fit_transform(t_anno)
v_anno = mm.transform(v_anno)

print("[YO] Data Loaded")

## git mlp and cnn
cnn = make_cnn(img_size, filters=(16, 32, 64), regress=False)
mlp = make_mlp(t_anno.shape[1], regress=False)

cat_input = concatenate([mlp.output, cnn.output])

x = Dense(4, activation="relu")(cat_input)
x = Dense(1, activation="linear")(x)

model = Model(inputs=[mlp.input, cnn.input], outputs=x)

print("[YO] Model Created")

opt = Adam(learning_rate=lr, decay=lr_d)
model.compile(
    loss="mean_absolute_percentage_error",
    optimizer=opt,
    metrics=["mean_absolute_percentage_error", "logcosh", "mean_squared_error"],
)

print("[YO] Training Started")

model.fit(
    [t_anno, t_img],
    t_brake,
    validation_data=([v_anno, v_img], v_brake),
    epochs=ep,
    batch_size=bs,
    callbacks=[tensorboard],
)

print("[YO] Training Done. Starting Predictions.")
preds = model.predict([v_anno, v_img])

diff = preds.flatten() - v_brake
percentDiff = (diff / v_brake) * 100
absPercentDiff = np.abs(percentDiff)
# print("Percent Diff= {}%".format(absPercentDiff))

mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

print("Mean= {}%".format(mean))
print("std= {}".format(std))
