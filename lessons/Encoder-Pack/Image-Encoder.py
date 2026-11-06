import datetime
import random
import time

import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras import Input
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model, Sequential


# np.random.seed(137)  # for reproducibility
def plotExampleImg(title, imageData, decodedData, Ydigits):
    fig, ax = plt.subplots(2, 10, figsize=(10, 2))
    fig.canvas.manager.set_window_title(title)
    for num in range(0, 10):
        numberImg = imageData[np.where(Ydigits == num)[0]]
        imgDecoded = decodedData[np.where(Ydigits == num)[0]]
        # Return random integers from 0 (inclusive) to high (exclusive).
        randomIndex = np.random.randint(0, numberImg.shape[0])
        plt.gray()
        ax[0][num].imshow(numberImg[randomIndex])
        ax[0][num].set_axis_off()
        ax[1][num].imshow(imgDecoded[randomIndex])
        ax[1][num].set_axis_off()

    plt.show()


# n_input = 784 # MNIST data input (img shape: 28*28)
def build_neural_network(input_len):
    model = Sequential()
    model.add(Input(shape=(input_len,)))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    # now model.output_shape == (None, 256)
    # note: `None` is the batch dimension.
    #
    # this bottleneck layer is the encoder output
    model.add(Dense(128, activation='relu', name='encoder_output'))
    # now model.output_shape == (None, 128)
    #
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    # now model.output_shape == (None, 256)

    model.add(Dense(input_len))

    # algorithm to train models: RMSProp
    # For a mean squared error regression problem
    model.compile(optimizer='rmsprop',
                  loss='mse',
                  metrics=['accuracy'])
    return model


def trainModel(model, Xtrain, epochs=20):
    global_start_time = time.time()
    model.fit(Xtrain, Xtrain, batch_size=256, epochs=epochs, verbose=0)
    sec = datetime.timedelta(seconds=int(time.time() - global_start_time))
    print('Training duration : ', str(sec))

    # evaluate all training set after trained
    scores = model.evaluate(Xtrain, Xtrain, verbose=0)
    print("Evaluate model: %s = %.4f" % (model.metrics_names[0], scores[0]))
    print("Evaluate model: %s = %.4f" % (model.metrics_names[1], scores[1] * 100))

    return model


if __name__ == "__main__":
    # Get MNIST Datasets
    (X_train, _), (X_test, Y_test) = mnist.load_data()
    _, heigh, width = X_train.shape
    total_pixel = heigh * width

    X_train = np.reshape(X_train, (-1, heigh * width))
    X_test = np.reshape(X_test, (-1, heigh * width))

    base_model = build_neural_network(X_train.shape[1])
    base_model = trainModel(base_model, X_train)
    print(base_model.summary())

    encoder_model = Model(inputs=base_model.inputs,
                          outputs=base_model.get_layer("encoder_output").output)
    print(encoder_model.summary())

    random_index = random.randrange(X_test.shape[0])
    encoded = encoder_model.predict(np.array([X_test[random_index]]), verbose=0)
    print("Encoding digit:", Y_test[random_index])
    print("Encoding (example):\n", encoded[0:10])

    X_decoded = base_model.predict(X_test, verbose=0)
    # reshape to [example, heigh, width]
    X_decoded = np.reshape(X_decoded, (-1, heigh, width))
    X_test = np.reshape(X_test, (-1, heigh, width))
    print("Plotting...")
    plotExampleImg("Encoding and Decoding", X_test, X_decoded, Y_test)
