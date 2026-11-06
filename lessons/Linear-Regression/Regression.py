from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter, LinearLocator

from scipy.stats import linregress
from sklearn import linear_model, preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

import Visualize as am

try:
    import tensorflow as tf
    from tensorflow.keras import Input
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import SGD
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False


def show_result(bias, weights, mse):
    print('Bias  = %s , Coefficients = %s , MSE = %s' % (bias, weights, mse))


def show_graph(X, Y, predict, title, xlabel, ylabel):
    plt.scatter(X, Y, color='b', label='data')
    plt.plot(X, predict, color='r', label='predict')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


def add_one(data_X):
    ones = np.ones((len(data_X), 1))
    # Add 1 vector to first column
    X = np.append(ones, data_X, axis=1)
    return X


# example 1: solve the normal equation directly
def predict_example1(data_X, Y):
    X = add_one(data_X)
    C = np.linalg.inv(X.T @ X) @ (X.T @ Y)  # Answer of coefficients
    # Finish training

    # Show model
    predict = X @ C  # prediction
    mse = mean_squared_error(Y, predict)
    b, w = C[0], C[1:]
    show_result(b, w, mse)
    return predict


# example 2: use sklearn library
def predict_example2(X, Y):
    regr = linear_model.LinearRegression()
    regr.fit(X, Y)
    # Finish training

    # Show model
    predict = regr.predict(X)  # prediction
    mse = mean_squared_error(Y, predict)
    show_result(regr.intercept_, regr.coef_, mse)
    return predict


# example 3: use TensorFlow 2 (GradientTape)
def predict_example3(X, Y):
    # Try to find values for weights and bias that compute FX = X * W + B
    num_feature = X.shape[1]
    W = tf.Variable(tf.random.truncated_normal((num_feature, 1)))
    B = tf.Variable(tf.random.truncated_normal((1, 1)))
    X = tf.constant(X.astype(np.float32))
    Y = tf.constant(Y.astype(np.float32))

    # Use gradient descent algorithm for optimizing
    learningRate = 0.01
    optimizer = tf.keras.optimizers.SGD(learning_rate=learningRate)

    # Try to fit the line
    mse, predict = None, None
    for _ in range(4000):
        with tf.GradientTape() as tape:
            FX = tf.matmul(X, W) + B
            # Minimize the mean squared errors.
            loss = tf.reduce_mean(tf.square(FX - Y))
        gradients = tape.gradient(loss, [W, B])
        optimizer.apply_gradients(zip(gradients, [W, B]))
        mse, predict = float(loss), FX.numpy()

    # Show model
    show_result(B.numpy(), W.numpy(), mse)
    return predict


# Neural network
# example 4: use Keras library (1 neuron)
def predict_example4(X, Y):
    num_feature = X.shape[1]
    model = Sequential()
    model.add(Input(shape=(num_feature,)))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer=SGD(learning_rate=0.01))
    model.fit(X, Y, epochs=4000, verbose=0)

    weights = model.layers[0].get_weights()
    w = weights[0]
    b = weights[1][0]

    # Show model
    predict = model.predict(X, verbose=0)  # prediction
    mse = mean_squared_error(Y, predict)
    show_result(b, w, mse)
    return predict


# example 5: use gradient descent algorithm (hard coded without library)
def isConvergence(value):  # check condition of convergence
    return np.absolute(value) <= 0.005  # set threshold


def isNan(value):
    return np.sum(np.isnan(value)) > 0


def plot_surface_error(data_X, Y):  # for visualization
    x_range = np.arange(-5, 15, 1)  # -5 < x-axis < 15 (increase 1 step)
    y_range = np.arange(-5, 15, 1)  # -5 < y-axis < 15 (increase 1 step)

    w0, w1 = np.meshgrid(x_range, y_range)
    Z = np.empty(w0.shape)  # same size as w0 and w1
    X = add_one(data_X)

    for i in range(len(x_range)):  # calculate mse of all (w0, w1) and save to Z
        for j in range(len(y_range)):
            C = np.array([[w0[i, j]], [w1[i, j]]])
            fx = X @ C
            Z[i, j] = mean_squared_error(Y, fx)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('w0 axis')
    ax.set_ylabel('w1 axis')
    ax.set_zlabel('MSE axis')
    surf = ax.plot_surface(w0, w1, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def predict_example5(data_X, Y):
    X = add_one(data_X)                        # add one to first column
    learningRate = 0.0001                      # initial learning rate
    C = np.zeros((data_X.shape[1] + 1, 1))     # initial coefficients
    assert C.shape == (data_X.shape[1] + 1, 1)

    FX_List, Acc_List, Loss_List = [], [], []  # declare empty lists

    def evaluate():
        FX = X @ C                       # predict prices
        score = 100 * r2_score(Y, FX)    # calculate R2 score for accuracy
        MSE = mean_squared_error(Y, FX)  # calculate mean squared error for loss
        # save it for visualization later
        FX_List.append(FX)
        Acc_List.append(score)
        Loss_List.append(MSE)

    # for visualization before training
    evaluate()

    step = 0
    while True:
        SLOPE = X.T @ (X @ C - Y)           # vector 2 x 1
        new_C = C - (learningRate * SLOPE)  # vector 2 x 1

        if isNan(SLOPE):
            print('Slope is NaN:', SLOPE)
            break

        C_update = np.copy(C)
        for i in range(0, len(SLOPE)):
            if isConvergence(SLOPE[i, 0]) == False:  # not convergence  # noqa: E712
                C_update[i, 0] = new_C[i, 0]

        C = C_update  # update new coefficients include bias

        if step % 100 == 0:  # for visualization later
            evaluate()
        step += 1

        # stop while_loop when all weights (coefficients) meet the convergence condition
        conv = isConvergence(SLOPE)
        if np.sum(conv) == len(SLOPE):
            break

    # Finish training
    print("Total step to learning:", step)

    # Show model
    evaluate()
    b, w = C[0, 0], C[1:, 0]
    show_result(b, w, Loss_List[-1:])

    # For visualization finally
    FX_List = np.reshape(FX_List, (-1, X.shape[0]))
    return FX_List, Acc_List, Loss_List


########## For one input ###################
# example 6: use numpy module (polyfit)
def predict_example6(X, Y):
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    w1, w0 = np.polyfit(X, Y, 1)
    # Finish training

    # Show model
    # use broadcasting rules in numpy to add matrix
    predict = w1 * X + w0  # prediction
    mse = mean_squared_error(Y, predict)
    show_result(w0, w1, mse)
    return predict


# example 7: use scipy module (linregress)
def predict_example7(X, Y):
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    slope, intercept, r, p, stderr = linregress(X, Y)
    # Show model
    predict = intercept + slope * X  # prediction
    mse = mean_squared_error(Y, predict)
    show_result(intercept, slope, mse)
    return predict


def prepare_dataset(csv_dataset, x_column_name, y_column_name, base_dir=""):
    # read csv file with pandas module
    df = pd.read_csv(join(base_dir, csv_dataset))

    print("First of 5 rows in Dataset")
    print(df.head())
    print("\nTail of 5 rows in Dataset")
    print(df.tail())

    train_X = df[x_column_name].values.reshape(-1, 1)  # X (Input) training set
    train_Y = df[y_column_name].values.reshape(-1, 1)  # Y (Output) training set
    return train_X, train_Y


######################
#### for test only ####
def test_one_input(X, train_Y, title, xlabel, ylabel):
    # Preprocessing data
    scaler_X = preprocessing.StandardScaler().fit(X)
    scaler_Y = preprocessing.StandardScaler().fit(train_Y)
    train_X = scaler_X.transform(X)
    train_Y = scaler_Y.transform(train_Y)

    assert train_X.shape == train_Y.shape

    print("\n+++++ Example 1++++")
    predict = predict_example1(train_X, train_Y)
    show_graph(X,
               scaler_Y.inverse_transform(train_Y),
               scaler_Y.inverse_transform(predict),
               title, xlabel, ylabel)

    print("\n+++++ Example 2++++")
    predict = predict_example2(train_X, train_Y)

    if HAS_TENSORFLOW:
        print("\n+++++ Example 3++++")
        predict = predict_example3(train_X, train_Y)

        print("\n+++++ Example 4++++")
        predict = predict_example4(train_X, train_Y)
    else:
        print("\n(Skipping examples 3-4: TensorFlow is not installed)")

    print("\n+++++ Example 5++++")
    predictList, accuracyList, lossList = predict_example5(train_X, train_Y)
    am.visualize(X,
                 scaler_Y.inverse_transform(train_Y),
                 scaler_Y.inverse_transform(predictList),
                 accuracyList, lossList, title=title)
    plot_surface_error(train_X, train_Y)

    print("\n+++++ Example 6++++")
    predict = predict_example6(train_X, train_Y)

    print("\n+++++ Example 7++++")
    predict = predict_example7(train_X, train_Y)  # noqa: F841


def test_polynomial(X, train_Y, title, xlabel, ylabel):
    # Preprocessing data
    scaler_X = preprocessing.StandardScaler().fit(X)
    scaler_Y = preprocessing.StandardScaler().fit(train_Y)
    X__ = scaler_X.transform(X)
    train_Y = scaler_Y.transform(train_Y)

    # Generate polynomial features (2 degree).
    degree = 2  # You can change to degree 3, 4, 5 and others here
    poly = PolynomialFeatures(degree)
    # output for degree 2 = [1, x, x^2]
    # output for degree 3 = [1, x, x^2, x^3]
    train_X = poly.fit_transform(X__)

    # remove 1 value from array (index 0)
    train_X = np.delete(train_X, [0], 1)

    assert train_X.shape == (len(train_X), degree)  # (xxx, degree)
    assert train_Y.shape == (len(train_Y), 1)       # (xxx, 1)

    print("\n+++++ Example 1++++")
    predict = predict_example1(train_X, train_Y)
    show_graph(X,
               scaler_Y.inverse_transform(train_Y),
               scaler_Y.inverse_transform(predict),
               title, xlabel, ylabel)

    print("\n+++++ Example 2++++")
    predict = predict_example2(train_X, train_Y)

    if HAS_TENSORFLOW:
        print("\n+++++ Example 3++++")
        predict = predict_example3(train_X, train_Y)

        print("\n+++++ Example 4++++")
        predict = predict_example4(train_X, train_Y)  # noqa: F841

    print("\n+++++ Example 5++++")
    predictList, accuracyList, lossList = predict_example5(train_X, train_Y)
    am.visualize(X,
                 scaler_Y.inverse_transform(train_Y),
                 scaler_Y.inverse_transform(predictList),
                 accuracyList, lossList, title=title)


def test_many_input(train_X, train_Y):
    scaler_X = preprocessing.StandardScaler().fit(train_X)
    scaler_Y = preprocessing.StandardScaler().fit(train_Y)
    train_X = scaler_X.transform(train_X)
    train_Y = scaler_Y.transform(train_Y)

    print("\n+++++ Example 1++++")
    predict = predict_example1(train_X, train_Y)

    print("\n+++++ Example 2++++")
    predict = predict_example2(train_X, train_Y)

    if HAS_TENSORFLOW:
        print("\n+++++ Example 3++++")
        predict = predict_example3(train_X, train_Y)

        print("\n+++++ Example 4++++")
        predict = predict_example4(train_X, train_Y)  # noqa: F841


if __name__ == '__main__':
    print("------- One input (one feature) --------")
    print("++++++++ Example food truck ++++++++")
    # Dataset.csv is the food-truck dataset from coursera:
    # https://www.coursera.org/learn/machine-learning taught by Andrew Ng
    X, train_Y = prepare_dataset('Dataset.csv',
                                 x_column_name='Input',
                                 y_column_name='Output')
    test_one_input(X, train_Y, title='Food truck',
                   xlabel='Population', ylabel='Profit')

    print("\n\n------- One input but many features (polynomial features) --------")
    print("++++++++ Example: Thailand population history ++++++++")
    X, train_Y = prepare_dataset('Population-Linear.csv',
                                 x_column_name='Year',
                                 y_column_name='Population')
    test_polynomial(X, train_Y, title='Thailand population',
                    xlabel='Years', ylabel='Population')

    print("\n\n----------- Many input ------------")
    print("++++++++++++ Example: California housing dataset +++++++++")
    # sklearn removed load_boston() (ethical concerns with the dataset);
    # the California housing dataset is the recommended replacement.
    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing()
    train_X, train_Y = housing.data, housing.target
    train_Y = np.reshape(train_Y, (len(train_Y), 1))  # shape: len(train_Y) x 1
    test_many_input(train_X, train_Y)
