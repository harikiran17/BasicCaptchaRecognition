import numpy as np
import tensorflow as tf
import keras

import preprocess as pre
import textdistance
import Levenshtein

characters, char_to_num, num_to_char = pre.generate_vocab()

def decode_batch_predictions(pred):
	input_len = np.ones(pred.shape[0]) * pred.shape[1]
	# Use greedy search. For complex tasks, you can use beam search
	results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
		:, :5
	]
	# Iterate over the results and get back the text
	output_text = []
	for res in results:
		res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
		output_text.append(res)
	return output_text

# evaluation metrics

def exact_match(a,b):
	m = [a[i] for i in range(len(a)) if a[i]==b[i]]
	return float(len(m)/len(a))

def jaccard(a,b):
	return float(textdistance.jaccard(a.split(),b.split()))

def levenshtein(a,b):
	return float(Levenshtein.ratio(a,b))

def word2vec(word):
	from collections import Counter
	from math import sqrt

	cw = Counter(word)
	# precomputes a set of the different characters
	sw = set(cw)
	# precomputes the "length" of the word vector
	lw = sqrt(sum(c*c for c in cw.values()))

	# return a tuple
	return cw, sw, lw

def cosdis(v1, v2):
	# which characters are common to the two words?
	common = v1[1].intersection(v2[1])
	# by definition of cosine distance we have
	return float(sum(v1[0][ch]*v2[0][ch] for ch in common)/v1[2]/v2[2])

