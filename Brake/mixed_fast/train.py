import os
import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

from models import gitCNN, gitMLP
from gitdata import gitImgs, gitAnno


## Check if gpu
# device_name = tf.test.gpu_device_name()
# if not device_name:
#     raise SystemError("GPU device not found")
# print("Found GPU at: {}".format(device_name))
# tf.summary.trace_on(profiler=True)


## Data dirs
data_dir = "/Users/Sean/Documents/VSC/Brake"
img_fold = data_dir + "/imgs"
txt_path = data_dir + "/annotations/North1009.txt"


## TensorBoard ish
NAME = "BrakeAI_{}".format(datetime.datetime.now())
logs = "logs/{}".format(NAME)
Tboard_callbacks = tf.keras.callbacks.TensorBoard(
    log_dir=logs, histogram_freq=1, profile_batch=15
)


## Hyper-parameters
ep = 10
lr = 1e-3
lr_d = 1e-3 / ep
img_size = 256
bs = 32


## Load Data
# IMGS
img_ds = gitImgs(img_fold, bs)

# ANNOTATIONS, brake and others
(train_anno, brake) = gitAnno(txt_path)
anno_ds = tf.data.Dataset.from_tensor_slices(train_anno)
brake_ds = tf.data.Dataset.from_tensor_slices(brake)

# Zip all 3, maybe see other ways to do this

DATASET = tf.data.Dataset.zip(((img_ds, anno_ds), (brake_ds)))

# DATASET_train = tf.data.Dataset.zip((img_ds, anno_ds))
# DATASET = tf.data.Dataset.zip((DATASET_train, brake_ds))

DATASET = DATASET.batch(bs)


## Git both CNN and MLP
mlp = gitMLP(3)
cnn = gitCNN(img_size, filters=(16, 32, 64))

cat_input = tf.keras.layers.concatenate([mlp.output, cnn.output])

x = tf.keras.layers.Dense(4, activation="relu")(cat_input)
x = tf.keras.layers.Dense(1, activation="linear")(x)

model = keras.Model(inputs=[cnn.input, mlp.input], outputs=x)


## BS keras model.fit ish
model.compile(
    loss="mean_absolute_percentage_error",
    optimizer=keras.optimizers.Adam(learning_rate=lr, decay=lr_d),
)

model.fit(DATASET, epochs=ep, callbacks=[Tboard_callbacks])


## Predictions. profiler?

