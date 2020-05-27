import os
import datetime
import tensorflow as tf
from tensorflow import keras
import numpy as np

from models_mlp import make_mlp

## Data dirs
data_dir = "/Users/Sean/Documents/VSC/Brake"
img_fold = data_dir + "/imgs"
txt_path = data_dir + "/annotations/North1009.txt"


## TensorBoard ish
NAME = "BrakeMLP_{}".format(datetime.datetime.now())
logs = "logs/{}".format(NAME)
Tboard_callbacks = tf.keras.callbacks.TensorBoard(
    log_dir=logs, histogram_freq=1, profile_batch=400
)


## Hyper-parameters
ep = 3
lr = 1e-2
lr_d = 1e-3 / ep
img_size = 64
bs = 32


## Load test (brake) and train (anno) data
anno_ds = np.loadtxt(txt_path, delimiter=",", dtype="float")
train = anno_ds[:, 1:3]
test = anno_ds[:, 0]


## Make DS
DATASET = tf.data.Dataset.from_tensor_slices((train, test))


## Git Model
mlp = make_mlp(train.shape[1], regress=True)
model = tf.keras.Model(inputs=mlp.input, outputs=mlp.output)


## Model.fit bs
model.compile(
    loss="mean_absolute_percentage_error",
    optimizer=keras.optimizers.Adam(learning_rate=lr, decay=lr_d),
)

model.fit(DATASET, epochs=ep, callbacks=[Tboard_callbacks])
