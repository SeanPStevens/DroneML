import tensorflow as tf
import numpy as np
import cv2


def imgs_np(annos, img_fold, resize_size):
    imgs = []
    img_size = (resize_size, resize_size)

    for anno in annos:
        img_dir = "{}/{}.jpg".format(img_fold, int(anno[0]))
        img = cv2.imread(img_dir)
        img = cv2.resize(img, img_size)
        imgs.append(img)

    return np.array(imgs)


def gitbytes(filepath):  # and label if needed
    raw_bytes = tf.io.read_file(filepath)

    return raw_bytes


def process_img(img_bytes):
    resolution = (64, 64)  # (256, 256)
    img = tf.io.decode_jpeg(img_bytes)
    img = tf.image.resize(img, resolution)
    img.set_shape((64, 64, 3))  # (256, 256, 3)
    img = img / 255.0  # - 0.5  ?

    # Img augmentations
    # img = tf.image.random_flip_left_right(img)
    # img = tf.image.random_flip_up_down(img)
    # img += tf.random.normal(img.shape, mean=0, stddev=.1)

    return img  # , tf.cast(label, tf.float32)


def img_tf_2(img_fold, bs):

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    # ds = tf.data.Dataset.list_files(img_fold)
    ds = tf.data.Dataset.list_files(str(img_fold + "/*.jpg"))
    # ds = ds.shuffle(NUM_TOTAL_IMGS)
    ds = ds.map(gitbytes, num_parallel_calls=AUTOTUNE)
    ds = ds.map(process_img, num_parallel_calls=AUTOTUNE)
    # ds = ds.batch(bs)
    return ds


def imgs_tf(img_fold):
    # bunch of decoding and shit
    bruh = img_fold
