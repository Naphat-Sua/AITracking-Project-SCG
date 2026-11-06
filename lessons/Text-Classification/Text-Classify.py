# thank you idea from: https://github.com/fchollet/keras/blob/master/examples/reuters_mlp.py

import datetime
import time

import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.decomposition import PCA

from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
	Activation,
	BatchNormalization,
	Conv1D,
	Dense,
	Dropout,
	Flatten,
	MaxPooling1D,
)

# Requirement
# pip install deepcut
from thai_dataset import load_dataset, load_dataset_unknown

MAX_WORDS = 5003


def sequences_to_binary_matrix(sequences, num_words=MAX_WORDS):
	"""Convert lists of word indexes to a binary bag-of-words matrix.

	Replacement for the removed keras Tokenizer.sequences_to_matrix(mode='binary'):
	output shape is (len(sequences), num_words) with 1.0 where a word index occurs.
	"""
	matrix = np.zeros((len(sequences), num_words))
	for row, seq in enumerate(sequences):
		for index in seq:
			if 0 <= index < num_words:
				matrix[row, index] = 1.0
	return matrix

def plotPCA2d(X, label_list, num_classes):
	estimator = PCA(n_components=2)
	Xpca = estimator.fit_transform(X)	
	assert np.shape(Xpca) == (np.shape(X)[0], 2)
	
	colors = ['red', 'green','blue', 'black' ]
	for label in range(0, num_classes): # 0 to 3
		label_list = np.array(label_list)	# convert to array with numpy
		AB = Xpca[np.where(label_list == label)[0]]	
		# seperate to a, b component 
		a = AB[:, 0]	
		b = AB[:, 1]
		plt.scatter(a, b, c=colors[label])
	plt.legend(np.arange(0,num_classes), loc='upper right')
	plt.xlabel('First Principal Component')
	plt.ylabel('Second Principal Component')
	plt.show()

def preprocessing(X_train, X_test, Y_train, Y_test, num_classes):
	print('Before convert of sequence words to binary matrix...')
	print('X_train shape:', np.shape(X_train))
	print('X_test shape:', np.shape(X_test))	
	
	print('Convert sequences of words (index) to binary matrix')
	# Return: numpy array of shape (len(sequences), num_words).
	X_train = sequences_to_binary_matrix(X_train, MAX_WORDS)
	X_test = sequences_to_binary_matrix(X_test, MAX_WORDS)
	print('X_train shape:', X_train.shape)
	print('X_test shape:', X_test.shape)	

	print('Convert class label (integers vector) to binary class matrix')
	Y_train = keras.utils.to_categorical(Y_train, num_classes)
	Y_test = keras.utils.to_categorical(Y_test, num_classes)
	print('Y_train shape:', Y_train.shape)
	print('Y_test shape:', Y_test.shape)
	return X_train, X_test, Y_train, Y_test

# Multilayer Perceptron (MLP)
def build_MLP(num_classes):
	model = Sequential()
	model.add(Input(shape=(MAX_WORDS,)))
	model.add(Dense(64))
	model.add(Activation('tanh'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes))
	model.add(Activation('softmax'))
	model.compile(loss='categorical_crossentropy',
				optimizer='adam',
				metrics=['accuracy'])
	return model

# Convolutional Neural Networks (CNN)
def build_CNN(num_classes):
	model = Sequential()
	model.add(Input(shape=(MAX_WORDS, 1)))
	model.add(Conv1D(filters=64, kernel_size=3, padding="same", activation='relu'))
	model.add(Conv1D(filters=64, kernel_size=3, padding="same", activation='relu'))
	model.add(MaxPooling1D(3))
	model.add(Conv1D(filters=64, kernel_size=3, padding="same", activation='relu'))
	model.add(Conv1D(filters=64, kernel_size=3, padding="same", activation='relu'))
	#model.add(GlobalAveragePooling1D())
	model.add(Flatten())
	model.add(Dense(units=10))
	model.add(Dropout(0.5))
	model.add(Dense(units=num_classes, activation='softmax'))
	model.compile(loss='categorical_crossentropy',
				optimizer='adam',
				metrics=['accuracy'])
	return model

# This model is not OK
def build_CNN2(num_classes):
	model = Sequential()
	model.add(Input(shape=(MAX_WORDS, 1)))
	model.add(Conv1D(filters=16, kernel_size=4, strides=1, padding="same"))
	model.add(BatchNormalization(trainable = True))
	model.add(Activation("relu"))
	model.add(Conv1D(filters=8, kernel_size=4, strides=1, padding="same"))	
	model.add(BatchNormalization(trainable = True))	
	model.add(Activation("relu"))
	model.add(Conv1D(filters=8, kernel_size=4, strides=1, padding="same"))	
	model.add(BatchNormalization(trainable = True))	
	model.add(Activation("relu"))
	model.add(Flatten())
	model.add(Dense(units = 10, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))
	model.compile(loss='categorical_crossentropy',
				optimizer='adam',
				metrics=['accuracy'])
	return model

def decode(Y_binary):
	# np.argmax: Returns the indices of the maximum values along an axis.	
	return np.array( [ np.argmax(list) for list in Y_binary] )
	
def train(model, X_train, X_test, Y_train, Y_test ):
	global_start_time = time.time()
	#automatic validation dataset 
	#model.fit(X_train, Y_train, batch_size=32, epochs=5, verbose=0, validation_split=0.1)
	model.fit(X_train, Y_train, batch_size=32, epochs=5, verbose=0,	validation_data=(X_test, Y_test))		
	sec = datetime.timedelta(seconds=int(time.time() - global_start_time))
	print ('Training duration : ', str(sec))
		
	# evaluate all training set after trained
	scores = model.evaluate(X_train, Y_train, verbose=0)
	print("Evalute model: %s = %.4f" % (model.metrics_names[0] ,scores[0]))
	print("Evalute model: %s = %.4f" % (model.metrics_names[1] ,scores[1]*100))	
	
	# for test only
	score = model.evaluate(X_test, Y_test, verbose=0)
	print('Test %s: %.4f' % (model.metrics_names[0], score[0]))
	print('Test %s: %.4f %%' % (model.metrics_names[1], score[1]*100))
	
	Ypredicted = model.predict(X_test, verbose=0)
	Yexpected = decode(Y_test)
	Ypredicted = decode(Ypredicted)
	print("Classification report")
	print(metrics.classification_report( Yexpected, Ypredicted))
	return model

def test(model, X_input):	
	predict = model.predict(X_input) 	# output shape is (1, number label)
	print("Predict: ", predict)
	index_label = np.argmax(predict[0])	
	return index_label
	
if __name__ == "__main__":	
	print('Loading data...')
	X_train, X_test, Y_train, Y_test = load_dataset()
	num_classes = np.max(Y_train) + 1
	print("Total class:", num_classes)

	print('Convert all data to binary vectorizing data')
	X_trainNew, X_testNew, Y_trainNew, Y_testNew = preprocessing(X_train, X_test, Y_train, Y_test, num_classes)
	print("Plot graph 2D (Principal component analysis (PCA))")
	plotPCA2d(X_trainNew, Y_train , num_classes) # use Y_train without vectorized

	print('\n+++++ Example: Building Multilayer Perceptron (MLP)  +++++')
	model_MPL = build_MLP(num_classes)
	print(model_MPL.summary())
	print('Training with MPL model...')
	train(model_MPL, X_trainNew, X_testNew, Y_trainNew, Y_testNew)
			
	print('\n+++++ Example: Convolutional Neural Networks (CNN) with Convolution1D +++++')
	model_CNN = build_CNN(num_classes)
	print(model_CNN.summary())
	print('Training with CNN model ...take a minute')	
	# For Convolutional 1D only, I reshaped input to (batch_size, steps, input_dim) 
	XX_trainNew = np.reshape(X_trainNew, (-1, MAX_WORDS, 1))
	XX_testNew = np.reshape(X_testNew, (-1, MAX_WORDS, 1))	
	train(model_CNN, XX_trainNew, XX_testNew, Y_trainNew, Y_testNew)
	
	# +++++++++++++++++++ For test only +++++++++++++++++++++++++++++++
	# label 0: 	"article", label 1: "encyclopedia", label 2: "news", label 3: "novel"
	label = ["Article", "Encyclopedia", "News", "Novel"]
	# I used this text file for test from : https://www.nectec.or.th/corpus/index.php?league=pm
	file_name = "Novel.txt.p"  # produced by running thai_dataset.py first
	name = file_name.replace('.p', '')
	content_index = load_dataset_unknown(file_name)

	# Convert sequences of words (index) to binary matrix
	content_binary = sequences_to_binary_matrix([content_index], MAX_WORDS)

	print("\n Test with data that never found: ", name)
	print('\nTesting for MPL model')
	index_label = test(model_MPL ,content_binary)
	print("Predict: '%s' is '%s'" % (name , label[index_label]))

	print('\nTesting for CNN model')
	cnn_input = np.reshape(content_binary, (-1, MAX_WORDS, 1))
	index_label = test(model_CNN, cnn_input)
	print("Predict: '%s' is '%s'" % (name , label[index_label]))
	
	
