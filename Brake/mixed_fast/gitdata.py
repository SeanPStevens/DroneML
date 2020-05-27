import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

size_img = 256


def gitbytes(filepath):  # and label if needed
    raw_bytes = tf.io.read_file(filepath)

    return raw_bytes


def process_img(img_bytes):
    resolution = (size_img, size_img)  # (256, 256)
    img = tf.io.decode_jpeg(img_bytes)
    img = tf.image.resize(img, resolution)
    img.set_shape((size_img, size_img, 3))  # (256, 256, 3)
    img = img / 255.0  # - 0.5  ?

    # Img augmentations
    # img = tf.image.random_flip_left_right(img)
    # img = tf.image.random_flip_up_down(img)
    # img += tf.random.normal(img.shape, mean=0, stddev=.1)

    return img  # , tf.cast(label, tf.float32)


def gitImgs(img_fold, bs):

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    # ds = tf.data.Dataset.list_files(img_fold)
    ds = tf.data.Dataset.list_files(str(img_fold + "/*.jpg"))
    # ds = ds.shuffle(NUM_TOTAL_IMGS)
    ds = ds.map(gitbytes, num_parallel_calls=AUTOTUNE)
    ds = ds.map(process_img, num_parallel_calls=AUTOTUNE)
    # ds = ds.batch(bs)
    return ds


def gitAnno(txt_dir):

    mm = MinMaxScaler()

    anno_ds = np.loadtxt(txt_dir, delimiter=",", dtype="float")

    brake = anno_ds[:, 0]
    train_anno = anno_ds[:, 1:4]

    brake = brake / brake.max()
    train_anno = mm.fit_transform(train_anno)

    return train_anno, brake
