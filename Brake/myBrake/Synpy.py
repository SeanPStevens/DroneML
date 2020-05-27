from gitdata import process_path, lots_imgs
import tensorflow as tf


data_dir = "C:\\Users\\Sean\\Documents\\VSC\\myBrake\\Data\\North45_3"
img_fold = "{}\\imgs".format(data_dir)
[train, val] = lots_imgs(img_fold, 32, 50000, 0.7)


## Data paths
# data_dir = "C:\\Users\\Sean\\Documents\\VSC\\myBrake\\Data\\North45_3"
# img_fold = "{}\\imgs".format(data_dir)

# list_ds = tf.data.Dataset.list_files(str(img_fold + "\\*.jpg"))

# for f in list_ds.take(5):
#     print(f.numpy())

# AUTOTUNE = tf.data.experimental.AUTOTUNE
# maped_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

# for image in maped_ds.take(1):
#     print("Image shape: ", image.numpy().shape)

# ds = maped_ds.batch(batch_size=32)
# ds = ds.prefetch(buffer_size=AUTOTUNE)

