import datetime
import random
import time

import numpy as np

from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model, Sequential


# np.random.seed(137)  # for reproducibility
def prepare_dataset(file_name):
    with open(file_name, encoding="utf8") as f:
        dictionary = f.read().lower().split()

    # to reduce duplicate words
    words = sorted(set(dictionary))
    len_words = len(words)
    print('total words:', len_words)

    # for converting a token to index
    char_indices = dict((c, i) for i, c in enumerate(words))
    # for converting index to a token
    indices_char = dict((i, c) for i, c in enumerate(words))

    print('Vectorization...')
    # One-hot encoding
    # the length of an encoded token vector == number of vocabulary
    Xtrain = np.zeros((len_words, len_words), dtype=bool)
    for t, w in enumerate(words):
        Xtrain[t, char_indices[w]] = 1

    return words, Xtrain, char_indices, indices_char


def trainModel(model, Xtrain, epochs):
    global_start_time = time.time()
    model.fit(Xtrain, Xtrain, batch_size=500, epochs=epochs, verbose=0)
    sec = datetime.timedelta(seconds=int(time.time() - global_start_time))
    print('Training duration : ', str(sec))

    # evaluate all training set after trained
    scores = model.evaluate(Xtrain, Xtrain, verbose=0)
    print("Evaluate model: %s = %.4f" % (model.metrics_names[0], scores[0]))
    print("Evaluate model: %s = %.4f" % (model.metrics_names[1], scores[1] * 100))

    return model


def build_neural_network(input_len):
    model = Sequential()
    model.add(Input(shape=(input_len,)))
    model.add(Dense(units=200, activation='tanh'))
    # now model.output_shape == (None, 200)
    # note: `None` is the batch dimension.
    #
    model.add(Dense(200, activation='tanh'))
    model.add(Dense(200, activation='tanh'))
    model.add(Dense(200, activation='tanh'))
    #
    # this bottleneck layer is the encoder output
    model.add(Dense(30, activation='tanh', name='encoder_output'))
    # now model.output_shape == (None, 30)
    #
    model.add(Dense(200, activation='tanh'))
    model.add(Dense(200, activation='tanh'))
    model.add(Dense(200, activation='tanh'))
    #
    model.add(Dense(input_len))

    # algorithm to train models: Adadelta
    model.compile(optimizer='adadelta',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    return model


if __name__ == "__main__":
    # reference: https://en.wikipedia.org/wiki/Most_common_words_in_English
    # Most common words in English: 100 words
    words, word_onehot, char_indices, indices_char = prepare_dataset('words_100.txt')
    base_model = build_neural_network(len(words))
    base_model = trainModel(base_model, word_onehot, epochs=500)
    print(base_model.summary())

    encoder_model = Model(inputs=base_model.inputs,
                          outputs=base_model.get_layer("encoder_output").output)
    print(encoder_model.summary())

    random_index = random.randrange(len(words))
    print("Testing with words: ", words[random_index])
    wd = word_onehot[random_index]
    wd = np.reshape(wd, (1, len(wd)))
    preds = base_model.predict(wd, verbose=0)
    index_prop = np.argmax(preds)
    word_predict = indices_char[index_prop]
    print("Output when decoding: ", word_predict)

    encoded = encoder_model.predict(wd, verbose=0)[0]
    print("Encoding:\n", encoded)
