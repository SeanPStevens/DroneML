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
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

import keras_multi_input.pyimagesearch.datasets as ds
import keras_multi_input.pyimagesearch.models as md
import datetime

NAME = "Test_House_{}".format(datetime.datetime.now())
logs = "logs/fit/{}".format(NAME)
tensorboard = TensorBoard(log_dir=logs)


# Get df
input_path = "/Users/Sean/Documents/VSC/Brake/annotations/HousesInfo.txt"

df = ds.load_house_attributes(input_path)

# get imgs
input_path_img = "/Users/Sean/Desktop/house_ds/Houses_Dataset"
images = ds.load_house_images(df, input_path_img)
images = images / 255.0


# test-train split
split = train_test_split(df, images, test_size=0.25)
(trainAttrX, testAttrX, trainImagesX, testImagesX) = split

maxPrice = trainAttrX["price"].max()
trainY = trainAttrX["price"] / maxPrice
testY = testAttrX["price"] / maxPrice

(trainAttrX, testAttrX) = ds.process_house_attributes(df, trainAttrX, testAttrX)
# print(trainAttrX)

mlp = md.create_mlp(trainAttrX.shape[1], regress=False)
cnn = md.create_cnn(64, 64, 3, regress=False)

combinedInput = concatenate([mlp.output, cnn.output])

x = Dense(4, activation="relu")(combinedInput)
x = Dense(1, activation="linear")(x)

model = Model(inputs=[mlp.input, cnn.input], outputs=x)

opt = Adam(lr=1e-4, decay=1e-4 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)
# train the model
print("[INFO] training model...")
model.fit(
    [trainAttrX, trainImagesX],
    trainY,
    validation_data=([testAttrX, testImagesX], testY),
    epochs=50,
    batch_size=8,
    callbacks=[tensorboard],
)
# make predictions on the testing data
print("[INFO] predicting house prices...")
preds = model.predict([testAttrX, testImagesX])

diff = preds.flatten() - testY
percentDiff = (diff / testY) * 100
absPercentDiff = np.abs(percentDiff)
print(absPercentDiff)

mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

print(mean)
print(std)
