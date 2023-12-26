from gc import callbacks
import os
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras.backend as K
from tensorflow.keras.layers import Layer

import preprocess as pre
import build_models as bm


batch_size = 32

img_width = 200
img_height = 50

SEED = 45632
np.random.seed(SEED)
tf.random.set_seed(SEED)
GPU_ID = 0

gpus = tf.config.list_physical_devices('GPU')
if(gpus):
    tf.config.set_visible_devices(gpus[GPU_ID], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[GPU_ID], True)

def get_data(input_images_path):
    
    images, labels = pre.read_images(input_images_path)

    characters, char_to_num, num_to_char = pre.generate_vocab()

    x_train, x_valid, y_train, y_valid = pre.split_data(images, labels)

    train_dataset, validation_dataset = pre.get_dataset(x_train, x_valid, y_train, y_valid)

    return train_dataset, validation_dataset, char_to_num, num_to_char

def get_callbacks(ckpt_path, logs_path):

    checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=ckpt_path,
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    save_weights_only=True
)

    csv_logger = keras.callbacks.CSVLogger(logs_path+"training.csv")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logs_path)

    return [checkpoint_callback, tensorboard_callback, csv_logger]

def train_model(model, train_dataset, validation_dataset, epochs, callbacks):
    history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    callbacks=callbacks)

    return history,model

def save_model(model, model_save_path):
    model.save(model_save_path)

def main():

    input_images_path = "./captcha_1_train/"
    ckpt_path = "./captcha_models/ckpt_crnn_full_captcha_1_32_200_50_gray_gru_attn"
    logs_path = "./captcha_models_logs/logs_crnn_full_captcha_1_32_200_50_gray_gru_attn"
    model_save_path = "./captcha_models/model_crnn_full_captcha_1_32_200_50_gray_gru_attn"

    train_dataset, validation_dataset, char_to_num, num_to_char = get_data(input_images_path)

    callbacks = get_callbacks(ckpt_path, logs_path)

    model = bm.build_model_5(img_width, img_height, char_to_num)

    history, model = train_model(model, train_dataset, validation_dataset, 200, validation_dataset)

    save_model(model, model_save_path)

main()