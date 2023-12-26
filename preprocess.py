import os
import numpy as np

from pathlib import Path
from tensorflow.keras import layers
import tensorflow as tf

def read_images(images_path):
    data_dir = Path("./datasets/captcha_1/captcha_1_train/")

    images = sorted(list(map(str, list(data_dir.glob("*.png")))))

    labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in images]
    labels = [l.split("_")[0] for l in labels]
    
    return images, labels
    
def generate_vocab():
    characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    characters = sorted(list(characters))

    # Mapping characters to integers
    char_to_num = layers.StringLookup(
        vocabulary=list(characters), mask_token=None
    )

    # Mapping integers back to original characters
    num_to_char = layers.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
    )

    return characters, char_to_num, num_to_char

characters, char_to_num, num_to_char = generate_vocab()

def split_data(images, labels, train_size=0.9, shuffle=True):
    size = len(images)
    indices = np.arange(size)
    if shuffle:
        np.random.shuffle(indices)
    train_samples = int(size * train_size)

    x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
    x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]

    return x_train, x_valid, y_train, y_valid

def encode_single_sample(img_path, label):

    img = tf.io.read_file(img_path)
    img = tf.io.decode_png(img, channels=1)
    #Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.transpose(img, perm=[1, 0, 2])
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))

    return {"image": img, "label": label}

def get_dataset(x_train, y_train, x_valid, y_valid, batch_size):
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = (
        train_dataset.map(
            encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
        )
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
    validation_dataset = (
        validation_dataset.map(
            encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
        )
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    return train_dataset, validation_dataset

