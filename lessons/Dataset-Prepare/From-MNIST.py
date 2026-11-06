import math

import numpy as np
from matplotlib import pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split

def plotExampleImg(title,imageData, Ydigits):
	fig = plt.figure()
	plt.gcf().canvas.manager.set_window_title(title)
	fig.set_facecolor('#FFFFFF')
	axList = []
	for position in range (1,11):
		ax = fig.add_subplot(2,5,position)
		ax.set_axis_off()
		axList.append(ax)		
		
	for num in range(0,10):
		numberImg = imageData[np.where(Ydigits == num)[0]]
		#Return random integers from 0 (inclusive) to high (exclusive).
		randomIndex = np.random.randint(0, numberImg.shape[0])		
		axList[num].imshow(numberImg[randomIndex])
	
	plt.axis('off')
	plt.show()
	
def example1():
	# fetch_mldata was removed from sklearn; fetch_openml is the replacement
	# (downloads MNIST on first run and caches it under ~/scikit_learn_data)
	mnist = datasets.fetch_openml("mnist_784", version=1, as_frame=False)
	x = mnist.data
	y = mnist.target.astype(int)
	print(mnist.keys())
	
	x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.33,
                                                        random_state=42)
		
	_, D = x_train.shape	
	W = int(math.sqrt(D))	
	assert D == 784 and W == 28
		
	imageData = x_train.reshape((-1, W, W))	# picture 28 x 28
	plotExampleImg("Example: 1", imageData, y_train)

from sklearn.datasets import load_digits
def example2():
	digits = load_digits()
	x = digits.data
	y = digits.target
	
	images = digits.images
	_, h, w = images.shape
	assert h == 8 and w == 8				# picture 8 x 8
	print(digits.keys())
	
	x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.33,
                                                        random_state=42)
														
	_, D = x_train.shape	
	W = int(math.sqrt(D))	
	assert D == 64 and W == 8
	
	imageData = x_train.reshape((-1, W, W))	# picture 8 x 8
	plotExampleImg("Example: 2", imageData, y_train)														

def example3():
	# tensorflow.contrib was removed in TensorFlow 2;
	# tf.keras.datasets.mnist is the replacement
	from tensorflow.keras.datasets import mnist

	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	assert x_train.shape == (60000, 28, 28)
	assert y_train.shape == (60000,)

	# take a batch of 100 images with 100 labels
	batch_X, batch_label = x_train[:100], y_train[:100]

	plotExampleImg("Example: 3", batch_X, batch_label)

if __name__ == "__main__":
	example1()
	example2()
	example3()
