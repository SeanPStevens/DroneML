import os
import datetime
import tensorflow as tf
from tensorflow import keras
import numpy as np

# from keras.callbacks import TensorBoard

from models_cnn import make_cnn
from CNNgitdata import imgs_np, imgs_tf, img_tf_2

## Data dirs
data_dir = "/Users/Sean/Documents/VSC/Brake"
img_fold = data_dir + "/imgs"
txt_path = data_dir + "/annotations/North1009.txt"

NAME = "BrakeCNN_{}".format(datetime.datetime.now())
logs = "logs/{}".format(NAME)
Tboard_callbacks = tf.keras.callbacks.TensorBoard(
    log_dir=logs, histogram_freq=1, profile_batch=400
)

## Check if gpu
# device_name = tf.test.gpu_device_name()
# if not device_name:
#     raise SystemError("GPU device not found")
# print("Found GPU at: {}".format(device_name))
# tf.summary.trace_on(profiler=True)

## BS paramaters
ep = 3
lr = 1e-2
lr_d = 1e-3 / ep
img_size = 64
bs = 32


## Load imgs and brake data
annos = np.loadtxt(txt_path, delimiter=",", dtype="float")
imgs = imgs_np(annos, img_fold, resize_size=img_size)
brake = annos[:, 1]


## Load imgs from tf loading pipeline
imgs_TFds = img_tf_2(img_fold, bs)
brake_ds = tf.data.Dataset.from_tensor_slices(brake)

ds = tf.data.Dataset.zip((imgs_TFds, brake_ds))
ds = ds.batch(bs)
# YOOOOOO this works!!!!
# Isn't exactly what you want, but model.fit is taking a
# zipped ds with imgs from my img_tf_2 loading pipeline and
# brake from a from_tensor_slices.
# allows for a lot more customization to the input scheme
# now see how you can apply prefetch, parallel loading, ...etc


## Zipped ds
# ds_imgs = tf.data.Dataset.from_tensor_slices(imgs)
# ds_b = tf.data.Dataset.from_tensor_slices(brake)

# ds = tf.data.Dataset.zip((ds_imgs, ds_b))
# ds = ds.batch(bs)
# this worked with model.fit()
# might just be because of np imgs to ds but will investigate


## Turn samples into tf datasets
# ds = tf.data.Dataset.from_tensor_slices((imgs, brake))
# ds = ds.batch(bs)
# this worked with model fit
# only problem is the need to have imgs in array form


## Model
cnn = make_cnn(img_size, filters=(16, 32, 64), regress=True)

model = keras.Model(inputs=cnn.input, outputs=cnn.outputs)

## Keras way
model.compile(
    loss="mean_absolute_percentage_error",
    optimizer=keras.optimizers.Adam(learning_rate=lr, decay=lr_d),
)

# model.fit(imgs, brake, epochs=ep, batch_size=bs)

model.fit(
    ds, epochs=ep, callbacks=[Tboard_callbacks],
)

# tf.summary.trace_export(name=NAME, profiler_outdir=logs)


## Custom Loop
