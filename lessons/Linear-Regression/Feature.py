from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.stats import linregress
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

import Visualize as am

try:
    import tensorflow as tf
    from tensorflow.keras import Input
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False


def show_result(w0, w1, mse):
    print('f(x) = %s + %sX , and MSE = %s' % (w0, w1, mse))
    print('Coefficients are w0 = %s, w1 = %s' % (w0, w1))


def add_one(data_X):
    ones = np.ones((len(data_X), 1))
    # Add 1 vector to first column
    X = np.append(ones, data_X, axis=1)
    return X


# Method 1: solve the normal equation
def train_method1(data_X, Y):
    X = add_one(data_X)
    C = np.linalg.inv(X.T @ X) @ (X.T @ Y)  # Answer of coefficients
    # Finish training

    # Show model
    FX = X @ C  # Linear line for prediction
    mse = mean_squared_error(Y, FX)
    w0, w1 = C
    show_result(w0, w1, mse)


# Method 2: use sklearn module
def train_method2(X, Y):
    regr = linear_model.LinearRegression()
    regr.fit(X, Y)
    # Finish training

    # Show model
    FX = regr.predict(X)  # Linear line for prediction
    mse = mean_squared_error(Y, FX)
    show_result(regr.intercept_, regr.coef_, mse)


# Method 3: use numpy module (polyfit)
def train_method3(X, Y):
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    w1, w0 = np.polyfit(X, Y, 1)
    # Finish training

    # Show model
    # use broadcasting rules in numpy to add matrix
    FX = w1 * X + w0  # Linear line for prediction
    mse = mean_squared_error(Y, FX)
    show_result(w0, w1, mse)


# Method 4: use scipy module (linregress)
def train_method4(X, Y):
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    slope, intercept, r, p, stderr = linregress(X, Y)

    # Show model
    FX = intercept + slope * X  # Linear line for prediction
    mse = mean_squared_error(Y, FX)
    show_result(intercept, slope, mse)


# Method 5: use TensorFlow 2 (GradientTape)
def train_method5(X, Y):
    # Try to find values for W_1 and W_0 that compute FX = W_1 * X + W_0
    W_1 = tf.Variable(tf.random.uniform([1], -1.0, 1.0))
    B = tf.Variable(tf.random.uniform([1], -1.0, 1.0))
    X = tf.constant(X.reshape(-1).astype(np.float32))
    Y = tf.constant(Y.reshape(-1).astype(np.float32))

    # Use gradient descent algorithm for optimizing
    learningRate = 0.01
    optimizer = tf.keras.optimizers.SGD(learning_rate=learningRate)

    # Try to fit the line
    mse = 0
    for _ in range(3000):
        with tf.GradientTape() as tape:
            FX = W_1 * X + B
            # Minimize the mean squared errors.
            loss = tf.reduce_mean(tf.square(FX - Y))
        gradients = tape.gradient(loss, [W_1, B])
        optimizer.apply_gradients(zip(gradients, [W_1, B]))
        mse = float(loss)

    # Show model
    show_result(B.numpy(), W_1.numpy(), mse)


# Method 6: use Keras library
def train_method6(X, Y):
    model = Sequential()
    model.add(Input(shape=(1,)))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.1))
    model.fit(X, Y, epochs=2000, verbose=0)

    weights = model.layers[0].get_weights()
    w1 = weights[0][0][0]
    w0 = weights[1][0]

    # Show model
    FX = w0 + w1 * X  # Linear line for prediction
    mse = mean_squared_error(Y, FX)
    show_result(w0, w1, mse)


# Method 7: use gradient descent algorithm (hard coded manually)
def isConvergence(value):  # check condition of convergence
    return np.absolute(value) <= 0.01  # set threshold


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
    ax.plot_surface(w0, w1, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                    linewidth=0, antialiased=False)
    plt.show()


def isNan(value):
    return np.sum(np.isnan(value)) > 0


def train_method7(data_X, Y):
    X = add_one(data_X)
    learningRate = 0.0001        # initial learning rate
    C = np.zeros((2, 1))         # initial coefficients

    FX_init = X @ C
    mse_init = mean_squared_error(Y, FX_init)
    print('First: f(x) = %s + %sx , and MSE = %s' % (C[0, 0], C[1, 0], mse_init))

    # save predicted values for visualization later
    FX_List = [FX_init]
    step = 0

    while True:
        SLOPE = X.T @ (X @ C - Y)           # vector 2 x 1
        new_C = C - (learningRate * SLOPE)  # vector 2 x 1

        if isNan(SLOPE):
            print('Slope is NaN:', SLOPE)
            break

        w0, w1 = C[0, 0], C[1, 0]
        s0, s1 = SLOPE[0, 0], SLOPE[1, 0]

        if isConvergence(s0) == False:  # noqa: E712
            w0 = new_C[0, 0]  # new w0

        if isConvergence(s1) == False:  # noqa: E712
            w1 = new_C[1, 0]  # new w1

        C = np.array([[w0], [w1]])  # update new coefficients

        if step % 100 == 0:  # for visualization later
            FX = X @ C
            FX_List = np.append(FX_List, FX)
        step += 1

        # stop while_loop when w0 and w1 meet the convergence condition
        if isConvergence(s0) and isConvergence(s1):
            break
    # Finish training
    print("Total step to learning:", step)

    # Show model
    FX_final = X @ C
    mse_final = mean_squared_error(Y, FX_final)
    w0, w1 = C
    show_result(w0, w1, mse_final)

    # for visualization
    FX_List = np.append(FX_List, FX_final)
    FX_List = np.reshape(FX_List, (-1, X.shape[0]))  # number of fx values x number of DatasetX
    return FX_List


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


#######################
## for test only
def run_testsuite(train_X, train_Y):
    print("\nSize of training set X: {}".format(train_X.shape))
    print("Size of training set Y: {}".format(train_Y.shape))

    print("\n+++++Show method 1++++")
    train_method1(train_X, train_Y)

    print("\n+++++Show method 2++++")
    train_method2(train_X, train_Y)

    print("\n+++++Show method 3++++")
    train_method3(train_X, train_Y)

    print("\n+++++Show method 4++++")
    train_method4(train_X, train_Y)

    if HAS_TENSORFLOW:
        print("\n+++++Show method 5++++")
        train_method5(train_X, train_Y)

        print("\n+++++Show method 6++++")
        train_method6(train_X, train_Y)
    else:
        print("\n(Skipping methods 5-6: TensorFlow is not installed)")

    # uncomment this if you want to show the contour of the error graph
    # plot_surface_error(train_X, train_Y)
    print("\n+++++Show method 7++++")
    FX_List = train_method7(train_X, train_Y)
    am.visualize(train_X, train_Y, FX_List)


if __name__ == '__main__':
    # Dataset.csv is the food-truck dataset from coursera:
    # https://www.coursera.org/learn/machine-learning taught by Andrew Ng
    train_X, train_Y = prepare_dataset(csv_dataset='Dataset.csv',
                                       x_column_name='Input',
                                       y_column_name='Output')
    run_testsuite(train_X, train_Y)
