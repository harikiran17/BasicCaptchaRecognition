
from distutils.errors import PreprocessError
import tensorflow as tf
import keras
import numpy as np
import cv2
import os

import random
from random import shuffle

import preprocess as pre
import post_process as post

characters, char_to_num, num_to_char = pre.generate_vocab()

def read_test_data(input_imgs_path):
	images_names =	[i for i in os.listdir(input_imgs_path) if i.endswith(".PNG") or i.endswith(".jpg") or i.endswith(".png")]


	labels = [l.split(".png")[0] for l in images_names]
	labels = [l.split("_")[0] for l in labels]
	idx_shuf=list(range(len(images_names)))
	shuffle(idx_shuf)
	images_names = [images_names[i] for i in idx_shuf]
	labels = [labels[i] for i in idx_shuf]

	return images_names, labels

def load_model(saved_model_path, ckpt_path):
	prediction_model = keras.models.load_model(saved_model_path)
	prediction_model = keras.models.Model(prediction_model.get_layer(name="image").input, prediction_model.get_layer(name="dense2").output)
	prediction_model.load_weights(ckpt_path)

	return prediction_model

def preprocess_image(img_path):
	image_raw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
	image_inp = image_raw[...,np.newaxis]
	image_inp = tf.convert_to_tensor(image_inp, tf.uint8)
	image_inp = tf.image.convert_image_dtype(image_inp,tf.float32)
	image_inp = tf.transpose(image_inp, perm=(1,0,2))
	image_inp = image_inp[np.newaxis,...]

	return image_inp

def main():
	input_imgs_path = "./captcha_1_test/"
	saved_model_path="./captcha_models/model_crnn_full_captcha_1_32_200_50_gray_gru_attn"
	ckpt_path ="./captcha_models/ckpt_crnn_full_captcha_1_32_200_50_gray_gru_attn"

	tot_em=0
	tot_jac=0
	tot_lev=0
	tot_cos=0

	c=0

	images_names, labels = read_test_data(input_imgs_path)

	prediction_model = load_model(saved_model_path, ckpt_path)

	for img in images_names:

		img_path = input_imgs_path+img
		image_inp = preprocess_image(img_path)

		pred = prediction_model(image_inp)

		output_text=post.decode_batch_predictions(pred)
		ot=output_text[0].replace("[UNK]","_")

		print(c,ot,labels[c],post.exact_match(ot,labels[c]),post.jaccard(ot,labels[c]),post.levenshtein(ot,labels[c]),post.cosdis(post.word2vec(ot),post.word2vec(labels[c])))

		tot_em+=post.exact_match(ot,labels[c])
		tot_jac+=post.jaccard(ot,labels[c])
		tot_lev+=post.levenshtein(ot,labels[c])
		tot_cos+=post.cosdis(post.word2vec(ot),post.word2vec(labels[c]))

		c+=1

	
	print("em = ",tot_em/len(images_names))
	print("jac = ",tot_jac/len(images_names))
	print("lev = ",tot_lev/len(images_names))
	print("cos = ",tot_cos/len(images_names))

main()