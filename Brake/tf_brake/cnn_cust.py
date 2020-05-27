import tensorflow as tf
import keras
import numpy as np
from keras import layers
from keras.models import Model
from keras.optimizers import Adam

from timeit import default_timer as timer

from gitdata2 import gitDS
from models import make_cnn

# Data paths
data_dir = "/Users/Sean/Documents/VSC/Brake"
img_fold = "{}/imgs".format(data_dir)
anno_path = "{}/annotations/North1009.txt".format(data_dir)

# Constant Parameters
batch_size = 8
epochs = 2
lr = 1e-3
lr_d = 1e-3 / epochs
img_size = 64

# Get CNN
cnn = make_cnn(256, filters=(16, 32, 64), regress=True)
model = Model(inputs=cnn.input, outputs=cnn.output)

# Instantiate an optimizer (Adam)
opt = Adam(learning_rate=lr, decay=lr_d)

# Instantiate the loss function (MAPE)
loss_fcn = keras.losses.MeanAbsolutePercentageError()

# Get DATASET
DATASET = gitDS(data_dir, batch_size)

# Create Training Loop
for epoch in range(epochs):  # Epoch Loop
    print("Epoch {}/{}".format(epoch, epochs))

    # Now loop over the DATASET contens
    for step, (img_train, brake_train) in enumerate(DATASET):

        # Open Gradient Tape to log operations?
        # forward pass?
        with tf.GradientTape() as tape:

            # Logits for (step-th) mini-batch
            logits = model(img_train)
            # not really sure wtf logits is, read lots

            # Loss for current mini-batch
            # Quantity to be minimized over training loop
            loss_output = loss_fcn(brake_train, logits)

        # Get the gradients from the grad tape for the trainable parameters
        grads = tape.gradient(loss_output, model.trainable_weights)

        # Use optimizer(Adam) to minimize loss though
        # some niiiiice gradient decent
        opt.apply_gradients(zip(grads, model.trainable_weights))

        # Output to interface every x batches.
        if step % batch_size == 0:
            print("TRAINING INFO")
            # ex/ TRAINING LOSS of one batch at step?
            # samples seen so far?

