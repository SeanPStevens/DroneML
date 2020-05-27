import os
import numpy as np
import absl
import cv2
import tensorflow as tf

# import keras
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator


def load_anno(txt_dir):
    df = []
    anno = np.loadtxt(txt_dir, delimiter=",", dtype="float")
    # df[:, 0] = anno
    return anno


def load_imgs(annos, imgs_dir, target_size):
    imgs = []

    for anno in annos:
        img_dir = "{}/{}.jpg".format(imgs_dir, int(anno[0]))
        img = cv2.imread(img_dir)
        img = cv2.resize(img, target_size)
        imgs.append(img)

    return np.array(imgs)


def csv_numeric(data_path):
    x = 1


def tf_numeric(data_path):
    ds = []
    ds = np.loadtxt(data_path, delimiter=",", dtype="float")
    maxBrake = ds[:, 1].max()
    t_brake = ds[:, 1] / maxBrake

    return t_brake


def resize_imgs(file_path):

    data_dir = "/Users/Sean/Documents/VSC/Brake"
    anno_path = "{}/annotations/North1009.txt".format(data_dir)
    brake = tf_numeric(anno_path)

    img = tf.io.read_file(file_path)

    img = tf.image.decode_jpeg(img, channels=3)

    img = tf.image.convert_image_dtype(img, tf.float32)

    # Need to experiment with making img size a input to the function
    img = tf.image.resize(img, [256, 256])

    return img, brake  # , tf.data.Dataset.from_tensor_slices(brake)


def gitImgs2(img_fold, bs):

    filenames = tf.data.Dataset.list_files(str(img_fold + "/*.jpg"))

    # Add test train split

    tf_dataset = filenames.map(
        resize_imgs, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    tf_dataset = tf_dataset.repeat()
    tf_dataset = tf_dataset.batch(8)
    return tf_dataset


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [256, 256])


def process_path(file_path):

    data_dir = "/Users/Sean/Documents/VSC/Brake"
    anno_path = "{}/annotations/North1009.txt".format(data_dir)
    brake = tf_numeric(anno_path)

    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img


def lots_imgs(img_fold, batch_size, total, split):
    list_ds = tf.data.Dataset.list_files(str(img_fold + "\\*.jpg"))
    n = total

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    maped_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    for image in maped_ds.take(1):
        print("Image shape: ", image.numpy().shape)

    ds = maped_ds.batch(batch_size=batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    train_size = int(split * n)
    val_size = int((1 - split) * n)
    # test_size = int(0.15 * n)

    train = ds.take(train_size)
    val = ds.skip(train_size)
    return [train, val]


def pro_img(img_fold, batch_size):

    data_dir = "/Users/Sean/Documents/VSC/Brake"
    img_fold = "{}/imgs".format(data_dir)
    anno_path = "{}/annotations/North1009.txt".format(data_dir)

    brake = tf_numeric(anno_path)

    # tf_dataset = gitImgs(img_fold, bs)

    list_ds = tf.data.Dataset.list_files(str(img_fold + "/*.jpg"))

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    maped_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    # for image, brake_num in maped_ds.take(1):
    #     print("Image shape: ", image.numpy().shape)
    #     print("Brake:", brake_num.numpy())

    maped_ds = maped_ds.repeat()
    maped_ds = maped_ds.batch(batch_size)

    return maped_ds
