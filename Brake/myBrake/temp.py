import tensorflow as tf
import numpy as np
from gitdata import load_anno, gitImgs2, process_path, tf_numeric
from gitdata2 import gitDS, gitAnno, gitImgs, csvAnno

data_dir = "/Users/Sean/Documents/VSC/Brake"
img_fold = "{}/imgs".format(data_dir)
anno_path = "{}/annotations/North1009.txt".format(data_dir)

### Put Test Scripts Below, if useful comment out test lines.

## Using zip to stitch num and img datasets
# probably won't work with model.fit
# brake = tf_numeric(anno_path)

# b_tf = tf.data.Dataset.from_tensor_slices(brake,)
# b_tf = b_tf.batch(8)

# print(b_tf.element_spec)

# for item in b_tf:
#     print(item.numpy())


# ds_filenames = tf.data.Dataset.list_files(str(img_fold + "/*.jpg"))

# AUTOTUNE = tf.data.experimental.AUTOTUNE
# imgs = ds_filenames.map(process_path, num_parallel_calls=AUTOTUNE)
# imgs = imgs.batch(8)

# print(imgs.element_spec)

# ds = tf.data.Dataset.zip((imgs, b_tf))

# print(ds.element_spec)

# for image, brake_num in ds.take(5):
#     print("Image shape: ", image.numpy().shape)
#     print("Brake:", brake_num.numpy())


## Using experimental csv read
# Didn't really work, but didn't try much
# brake = csvAnno(anno_path)

# b_tf = tf.data.Dataset.from_tensor_slices(brake,)
# b_tf = b_tf.batch(8)

# ds_filenames = tf.data.Dataset.list_files(str(img_fold + "/*.jpg"))

# AUTOTUNE = tf.data.experimental.AUTOTUNE
# imgs = ds_filenames.map(process_path, num_parallel_calls=AUTOTUNE)

# ds = tf.data.Dataset.zip((imgs, b_tf))

# for image, brake_num in ds.take(5):
#     print("Image shape: ", image.numpy().shape)
#     print("Brake:", brake_num.numpy())
