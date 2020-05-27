import tensorflow as tf
import numpy as np


def Load_Resize(img_path):
    # Read file contents
    img = tf.io.read_file(img_path)

    # Convert raw to tf dense string
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    # Resize
    img = tf.image.resize(img, [256, 256])

    # Any img Augmentations put in this section

    return img


def gitImgs(img_folder, BatchSize):

    # Create Path list (tf array? tensor? compressed string tensor? dataset object? for each jpg in img_folder
    img_filenames = tf.data.Dataset.list_files(str(img_folder + "/*.jpg"))

    # For each img path in
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    imgs = img_filenames.map(Load_Resize, num_parallel_calls=AUTOTUNE)

    return imgs


def gitAnno(anno_path):

    # Initialize array to store shit in
    ds = []

    # Load text file into numpy array
    ds = np.loadtxt(anno_path, delimiter=",", dtype="float")

    # Normalize data
    maxBrake = ds[:, 1].max()
    normBrake = ds[:, 1] / maxBrake

    # Make a Tensorflow dataset containing selected numerical data
    annotations = tf.data.Dataset.from_tensor_slices(normBrake)

    return annotations


def csvAnno(anno_path):
    ds = tf.data.experimental.make_csv_dataset(
        anno_path, header=False, select_columns=[1], batch_size=8
    )
    return ds


def gitDS(data_dir, BatchSize):

    # Initialize some paths
    img_fold = "{}/imgs".format(data_dir)
    anno_path = "{}/annotations/North1009.txt".format(data_dir)

    # Load Numerical motion data
    anno = gitAnno(anno_path)
    # anno = anno.batch(BatchSize)

    # Load Image Data
    imgs = gitImgs(img_fold, BatchSize)
    # imgs = imgs.batch(BatchSize)

    # Combine Datasets to make one dataset
    DATASET = tf.data.Dataset.zip((imgs, anno))

    # Any dataset augmentations here
    DATASET = DATASET.repeat()
    DATASET = DATASET.batch(BatchSize)  # .repeat()
    return DATASET  # , anno, imgs

