"""A 2-layer neural network that learns the logic `(X1 or X2) xor X3`.

Written with TensorFlow 2 (GradientTape). Training only runs when this file is
executed directly, so the helpers below can be imported and reused.
"""
import os

import numpy as np
import pandas as pd
import tensorflow as tf

CSV_DIR = os.path.dirname(os.path.abspath(__file__))  # folder of this script
CSV_FILE = 'Layer.csv'

L = 5  # Number of nodes (neurons) of layer1
M = 1  # Number of nodes (neurons) of layer2


def load_dataset(csv_dir=CSV_DIR, csv_file=CSV_FILE):
    """Read the truth table and return (dataframe, X, Y)."""
    df = pd.read_csv(os.path.join(csv_dir, csv_file), dtype=np.float32)
    data_X = df[['X1', 'X2', 'X3']].values             # training input
    # correct answers will go here
    data_Y = df['X1 or X2 xor X3'].values.reshape(-1, 1)  # convert to the vector
    return df, data_X, data_Y


def build_variables(total_features, seed=0):
    """Declare the weights and biases that get updated when training."""
    tf.random.set_seed(seed)
    W1 = tf.Variable(tf.random.truncated_normal([total_features, L], stddev=0.1))
    B1 = tf.Variable(tf.zeros([L]))
    W2 = tf.Variable(tf.random.truncated_normal([L, M], stddev=0.1))
    B2 = tf.Variable(tf.zeros([M]))
    return W1, B1, W2, B2


def forward(X, W1, B1, W2, B2):
    """2 fully connected layers."""
    L1 = tf.nn.sigmoid(tf.matmul(X, W1) + B1)      # output: samples x L
    return tf.nn.sigmoid(tf.matmul(L1, W2) + B2)   # prediction output: samples x M


def train(data_X, data_Y, learning_rate=0.1, max_steps=30000, seed=0):
    """Backpropagation training; stops early at 100% accuracy."""
    X = tf.constant(data_X)
    Y_ = tf.constant(data_Y)

    variables = build_variables(data_X.shape[1], seed=seed)
    W1, B1, W2, B2 = variables

    # use gradient descent algorithm to update weight and bias (W1, W2, B1, B2)
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    for step in range(max_steps):
        with tf.GradientTape() as tape:
            Predict = forward(X, *variables)
            # loss = Predict - correct answers
            loss = tf.reduce_mean(tf.square(Predict - Y_))

        gradients = tape.gradient(loss, list(variables))
        optimizer.apply_gradients(zip(gradients, list(variables)))

        # If pred is more than 0.5, the answer is 1
        # If pred is less than 0.5, the answer is 0
        # In python, True is 1 and False is 0
        pred = 1 * (Predict.numpy() > 0.5)

        # accuracy of the trained model
        accuracy = np.mean(1 - np.abs(pred - data_Y)) * 100
        if accuracy == 100:  # finish
            print('\nTraining accuracy: %.1f%%' % accuracy)
            print('Finish learning at step: %d' % step)
            break

        if step % 3000 == 0:  # for debug
            print('\nTraining accuracy: %.1f%%' % accuracy)
            print('Loss at step %d: %f' % (step, float(loss)))

    return [v.numpy() for v in variables]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def predict(new_X, w1, b1, w2, b2):
    """Run one input through the trained weights, in plain numpy."""
    output_layer1 = sigmoid(np.matmul(new_X, w1) + b1)
    output_layer2 = sigmoid(np.matmul(output_layer1, w2) + b2)
    return output_layer1, output_layer2


def main():
    df, data_X, data_Y = load_dataset()
    w1, b1, w2, b2 = train(data_X, data_Y)

    print('\nAll input/output dataset')
    print(df)

    X1, X2, X3 = df['X1'], df['X2'], df['X3']

    # (X1 or X2) xor X3
    result = np.logical_xor(np.logical_or(X1, X2), X3)
    print('\nLogic result: X1 or X2 xor X3')
    print(result)

    new_X = [0, 1, 1]  # input for testing
    output_layer1, output_layer2 = predict(new_X, w1, b1, w2, b2)

    print('\nFor testing')
    print('\nInput:', new_X)
    print('\nWeight for layer 1:\n', w1)
    print('\nWeight for layer 2:\n', w2)
    print('\nOutput values of layer 1:\n', output_layer1)
    print('\nOutput values of layer 2:\n', output_layer2)

    answer = 1 * (output_layer2 > 0.5)
    print('\nAnswer is ', answer[0])


# see DNNClassifier, it looks much easier
# https://www.tensorflow.org/api_docs/python/tf/estimator/DNNClassifier
if __name__ == '__main__':
    main()
