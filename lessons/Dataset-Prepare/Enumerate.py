"""Enumerate the 43 layers of the pre-trained VGG-19 model and dump the
weights/biases to pickle files.

Nothing happens at import time: call load_vgg() (or run this file) to read the
.mat model, which is a large download.
"""
import os
import pickle

import numpy as np
import scipy.io

# Use VGG-Network: https://arxiv.org/abs/1409.1556 (paper)
# Download imagenet-vgg-verydeep-19.mat from
# http://www.vlfeat.org/matconvnet/pretrained/ and set VGG_PATH to its location
# (or export the VGG_PATH environment variable).
VGG_PATH = os.environ.get("VGG_PATH", "imagenet-vgg-verydeep-19.mat")


def load_vgg(vgg_path=None, verbose=True):
	"""Load the VGG-19 .mat file.

	Returns (W, B, meanColor): the weights and biases keyed by layer name, and
	the average (R, G, B) colour of the training set.
	"""
	vgg_path = vgg_path or VGG_PATH
	if not os.path.exists(vgg_path):
		raise SystemExit(
			"VGG model file not found: %s\n"
			"Download imagenet-vgg-verydeep-19.mat from "
			"http://www.vlfeat.org/matconvnet/pretrained/ and set the VGG_PATH "
			"environment variable to its location." % vgg_path
		)

	dataVGG = scipy.io.loadmat(vgg_path)
	dataLayer = dataVGG['layers'][0]
	assert dataLayer.shape == (43,)  # all layers

	# get color mean
	mean = dataVGG['normalization'][0][0][0]
	assert mean.shape == (224, 224, 3)

	# the average color: Red, Green, Blue should be [123.68, 116.779, 103.939]
	meanColor = np.mean(mean, axis=(0, 1))

	W, B = {}, {}
	for layer in dataLayer:
		dd = layer[0][0]
		if len(dd) <= 2:
			# 'relu layer' names
			if verbose:
				print(dd[1])
		elif dd[3] == 'pool':
			# 'max pool' layer names
			if verbose:
				print(dd[3])
		else:
			name = dd[3][0]
			weights, bias = dd[0][0]
			# names of convolutional and fully connected layers
			if verbose:
				print(name, " | weights:", weights.shape, " | bias:", bias.shape)
			W[name] = weights
			B[name] = bias

	return W, B, meanColor


# print all layers (43) in the VGG 19 model
"""
layer name		Size's weights			Size's bias
['conv1_1'] 	(3, 3, 3, 64) 			(1, 64)
['relu1_1']
['conv1_2'] 	(3, 3, 64, 64) 			(1, 64)
['relu1_2']
['pool']
['conv2_1'] 	(3, 3, 64, 128) 		(1, 128)
['relu2_1']
['conv2_2'] 	(3, 3, 128, 128) 		(1, 128)
['relu2_2']
['pool']
['conv3_1'] 	(3, 3, 128, 256) 		(1, 256)
['relu3_1']
['conv3_2'] 	(3, 3, 256, 256) 		(1, 256)
['relu3_2']
['conv3_3'] 	(3, 3, 256, 256) 		(1, 256)
['relu3_3']
['conv3_4'] 	(3, 3, 256, 256) 		(1, 256)
['relu3_4']
['pool']
['conv4_1'] 	(3, 3, 256, 512) 		(1, 512)
['relu4_1']
['conv4_2'] 	(3, 3, 512, 512) 		(1, 512)
['relu4_2']
['conv4_3'] 	(3, 3, 512, 512) 		(1, 512)
['relu4_3']
['conv4_4'] 	(3, 3, 512, 512) 		(1, 512)
['relu4_4']
['pool']
['conv5_1'] 	(3, 3, 512, 512) 		(1, 512)
['relu5_1']
['conv5_2'] 	(3, 3, 512, 512) 		(1, 512)
['relu5_2']
['conv5_3'] 	(3, 3, 512, 512) 		(1, 512)
['relu5_3']
['conv5_4'] 	(3, 3, 512, 512) 		(1, 512)
['relu5_4']
['pool']
['fc6'] 		(7, 7, 512, 4096) 		(1, 4096)
['relu6']
['fc7'] 		(1, 1, 4096, 4096) 		(1, 4096)
['relu7']
['fc8'] 		(1, 1, 4096, 1000) 		(1, 1000)
['prob']
"""


# dump all weights and bias
def dumpData(W, B):
	with open("weights.p", "wb") as f:
		pickle.dump(W, f)
	with open("bias.p", "wb") as f:
		pickle.dump(B, f)

	with open("weights.p", "rb") as f:
		WW = pickle.load(f)
	with open("bias.p", "rb") as f:
		BB = pickle.load(f)

	# Testing
	print("\nAfter reading from the pickle file")
	print("\nShape of weights")
	for key, value in sorted(WW.items()):
		print(key, value.shape)

	print("\nShape of bias")
	for key, value in sorted(BB.items()):
		print(key, value.shape)


if __name__ == '__main__':
	W, B, meanColor = load_vgg()
	print("\nAverage colour (R, G, B):", meanColor)
	# dumpData(W, B)
