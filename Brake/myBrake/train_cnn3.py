import datetime
import os

import tensorflow as tf

import keras

# from tensorflow import keras
import numpy as np
from keras.callbacks import TensorBoard
from keras.layers import concatenate
from keras.layers.core import Dense
from keras.models import Model
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from gitdata2 import gitDS
from models import make_cnn

from timeit import default_timer as timer

# Data paths
data_dir = "/Users/Sean/Documents/VSC/Brake"
img_fold = "{}/imgs".format(data_dir)
anno_path = "{}/annotations/North1009.txt".format(data_dir)

# Some parameters
bs = 8
ep = 2
lr = 1e-3
lr_d = 1e-3 / ep
img_size = 64

# Fuckin TensorBoard
NAME = "CNN_1_{}".format(datetime.datetime.now())
logs = "Brake/myBrake/logs/{}".format(NAME)
tensorboard = TensorBoard(log_dir=logs)

print("[YOOO] Starting Loading Dataset")
time1 = timer()

# Load DATASET
# DATASET = gitDS(data_dir, bs)
(DATASET, anno, imgs) = gitDS(data_dir, bs)

time2 = timer()
print("[YOOO] Dataset Loaded: {} sec".format(time2 - time1))

# Git CNN model
cnn = make_cnn(64, filters=(16, 32, 64), regress=True)

model = Model(inputs=cnn.input, outputs=cnn.output)

opt = Adam(learning_rate=lr, decay=lr_d)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

# print("[YOOO] Model Created")

# Fit Model to Data
print("[YOOO] Training Started")

# DATASET.ndim = 1009
# BatchDataset.shape = (8, 256, 256, 3)
model.fit(DATASET, epochs=ep)

# Predictions
print("[YOOO] Training Done. Starting Predictions.")
