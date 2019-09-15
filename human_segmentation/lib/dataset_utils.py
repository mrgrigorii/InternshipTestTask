import os
import functools

import tensorflow as tf
import lib.image_augmentation as img_aug


def get_baseline_dataset(filenames,
                         labels,
                         preproc_fn=functools.partial(img_aug.augment),
                         threads=4,
                         batch_size=8,
                         shuffle=True):
    num_x = len(filenames)
    # Create a dataset from the filenames and labels
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    # Map our preprocessing function to every element in our dataset, taking
    # advantage of multithreading
    dataset = dataset.map(_process_pathnames, num_parallel_calls=threads)
    if preproc_fn.keywords is not None and 'resize' not in preproc_fn.keywords:
        assert batch_size == 1, "Batching images must be of the same size"

    dataset = dataset.map(preproc_fn, num_parallel_calls=threads)

    if shuffle:
        dataset = dataset.shuffle(num_x)

    # It's necessary to repeat our data for all epochs
    dataset = dataset.repeat().batch(batch_size)
    return dataset


def get_images_filenames(x_path, y_path):
    images_p = os.listdir(x_path)
    images = []
    masks = []
    for image_p in images_p:
        name = image_p.split('.')[0]
        images.append(f'{x_path}/{name}.jpg')
        if y_path:
            masks.append(f'{y_path}/{name}.png')
    return images, masks


def _process_pathnames(fname, label_path):
    # We map this function onto each pathname pair
    img_str = tf.io.read_file(fname)
    img = tf.image.decode_jpeg(img_str, channels=3)

    label_img_str = tf.io.read_file(label_path)
    label_img = tf.image.decode_png(label_img_str)
    return img, label_img
