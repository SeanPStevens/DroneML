import tensorflow as tf
from gitdata import load_anno, gitImgs, process_path, tf_numeric

data_dir = "/Users/Sean/Documents/VSC/Brake"
img_fold = "{}/imgs".format(data_dir)
anno_path = "{}/annotations/North1009.txt".format(data_dir)

brake = tf_numeric(anno_path)
brake_tf = tf.constant(brake)

# tf_dataset = gitImgs(img_fold, bs)

list_ds = tf.data.Dataset.list_files(str(img_fold + "/*.jpg"))

AUTOTUNE = tf.data.experimental.AUTOTUNE
maped_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

for image, brake_num in maped_ds.take(1):
    print("Image shape: ", image.numpy().shape)
    print("Brake:", brake_num.numpy())

maped_ds = maped_ds.repeat()
maped_ds = maped_ds.batch(8)

test_b = next(iter(maped_ds))
print(test_b)
