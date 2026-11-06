from os.path import dirname, join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import Visualize as am


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# show the sigmoid function
def show_sigmoid():
    x = np.arange(-50, 50, 1)
    y = sigmoid(x)
    plt.plot(x, y, 'r-')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()


# Example: show cost function
def cost_function(sg, label):
    # if label == 1: return -np.log(sg)
    # if label == 0: return -np.log(1-sg)
    return -label * np.log10(sg) - (1 - label) * np.log10(1 - sg)


def show_cost_function():
    # ตัวอย่างทำนายถูกต้อง
    # ทำนายถูก อยู่เหนือกราฟ decision boundary
    z1 = np.arange(0, 50, 0.01)
    p1 = sigmoid(z1)  # ให้ label เป็น 1
    c1 = cost_function(p1, 1)
    plt.plot(z1, c1, 'r-')

    # ทำนายถูก อยู่ต่ำกว่ากราฟ decision boundary
    z0 = np.arange(-50, 0, 0.01)
    p0 = sigmoid(z0)  # ให้ label เป็น 0
    c0 = cost_function(p0, 0)
    plt.plot(z0, c0, 'b-')

    plt.xlabel('sigmoid')
    plt.ylabel('error')
    plt.show()

    # ตัวอย่างทำนายผิดหมด
    # ทำนายผิด อยู่เหนือกราฟ แต่จริงๆ ต้องอยู่ใต้กราฟ decision boundary
    z1 = np.arange(0, 50, 0.1)
    p1 = sigmoid(z1)  # ให้ label เป็น 1 ที่ถูกต้องแล้ว ต้องเป็น 0
    c1 = cost_function(p1, 0)
    plt.plot(z1, c1, 'b-')

    # ทำนายผิด อยู่ต่ำกว่ากราฟ แต่จริงๆ ต้องอยู่เหนือกราฟ decision boundary
    z0 = np.arange(-50, 0, 0.1)
    p0 = sigmoid(z0)  # ให้ label เป็น 0 ที่ถูกต้องแล้ว ต้องเป็น 1
    c0 = cost_function(p0, 1)
    plt.plot(z0, c0, 'r-')

    plt.xlabel('sigmoid')
    plt.ylabel('error')
    plt.show()


# The exam-scores dataset (from Andrew Ng's machine learning course):
# two exam scores per student and a 0/1 admission label.
CSV_DATASET = join(dirname(__file__), '..', 'Dataset', 'Norm-Classification.csv')

x_column_names = ['X1', 'X2']
y_column_name = 'Y'


def load_dataset(csv_dataset=CSV_DATASET):
    df = pd.read_csv(csv_dataset)
    # normalize the column names used throughout this example
    df = df.rename(columns={'X': 'X1', 'Y': 'X2', 'Label': 'Y'})
    return df


# separate data in DataFrame:
# If column Y = 1, it is class A
# If column Y = 0, it is class B
def seperateClass(df):
    dfClassA = df.query('%s == 1' % y_column_name)
    dfClassB = df.query('%s == 0' % y_column_name)
    return dfClassA, dfClassB


def splitFeature(dfClass):
    nameX1, nameX2 = x_column_names
    return dfClass[nameX1], dfClass[nameX2]


def plot2Class(classA, classB, X1, FX):
    # plot class A
    X1A, X2A = splitFeature(classA)
    plt.plot(X1A, X2A, 'ro')  # red circle

    # plot class B
    X1B, X2B = splitFeature(classB)
    plt.plot(X1B, X2B, 'bo')  # blue circle

    # plot the decision boundary
    plt.plot(X1, FX, 'k-')  # black line
    plt.show()


def add_one(data_X):
    data_X = data_X.copy()
    data_X.insert(0, 'X0', 1)
    print("\nSize of matrix after adding 1 to first column:", data_X.shape)
    return data_X.values


# Method: use gradient descent algorithm
def isConvergence(value):  # check condition of convergence
    return np.absolute(value) <= 0.0001  # set threshold


def isNan(value):
    return np.sum(np.isnan(value)) > 0


def mean_cost(SX, Y, eps=1e-15):
    # clip predictions away from exactly 0/1 so log10 never sees zero
    SX = np.clip(np.asarray(SX).reshape(-1), eps, 1 - eps)
    return np.mean(cost_function(SX, np.asarray(Y).reshape(-1)))


def train_method(data_X, Y, learningRate=0.1, max_steps=200000):
    X = add_one(data_X)
    C = np.zeros((3, 1))              # initial coefficients (vector)

    SX_init = sigmoid(X @ C)          # multiply matrix
    loss = mean_cost(SX_init, Y)
    print('\nFirst: %s + %sX1 +%sX2 , and loss = %s' % (C[0, 0], C[1, 0], C[2, 0], loss))

    # save coefficients for visualization later
    C_List = [C]
    step = 0

    while True:
        SX = sigmoid(X @ C)                 # multiply matrix
        SLOPE = X.T @ (SX - Y)              # vector 3 x 1
        new_C = C - (learningRate * SLOPE)  # vector 3 x 1

        if isNan(SLOPE):
            print('Slope is NaN:', SLOPE)
            break

        w0, w1, w2 = C[0, 0], C[1, 0], C[2, 0]
        s0, s1, s2 = SLOPE[0, 0], SLOPE[1, 0], SLOPE[2, 0]

        if isConvergence(s0) == False:  # noqa: E712
            w0 = new_C[0, 0]  # new w0

        if isConvergence(s1) == False:  # noqa: E712
            w1 = new_C[1, 0]  # new w1

        if isConvergence(s2) == False:  # noqa: E712
            w2 = new_C[2, 0]  # new w2

        C = np.array([[w0], [w1], [w2]])  # update new coefficients

        if step % 20 == 0:  # for visualization later
            C_List = np.append(C_List, C)
        step += 1

        # stop while_loop when w0, w1 and w2 meet the convergence condition
        if np.sum(isConvergence(SLOPE)) == len(SLOPE):
            break

        # safety net: never loop forever if the learning rate diverges
        if step >= max_steps:
            print("Stopped at max_steps =", max_steps)
            break

    # Finish training

    # Show model
    SX_final = sigmoid(X @ C)
    loss = mean_cost(SX_final, Y)
    print('\nFinal: %s + %sX1 +%sX2 , and loss = %s' % (C[0, 0], C[1, 0], C[2, 0], loss))

    C_List = np.append(C_List, C)
    C_List = np.reshape(C_List, (-1, X.shape[1]))  # num iterations x features
    return C_List


def getDecisionFunc(X1, C_List):
    FX_List = []
    for C in C_List:
        w0, w1, w2 = C
        FX = (w0 + w1 * X1) / -w2
        FX_List = np.append(FX_List, FX)

    FX_List = np.reshape(FX_List, (-1, X1.shape[0]))  # num iterations x num samples
    return FX_List


if __name__ == "__main__":
    show_sigmoid()
    show_cost_function()

    df = load_dataset()

    print("\nFirst of 5 rows in Dataset")
    print(df.head())
    print("\nTail of 5 rows in Dataset")
    print(df.tail())
    print("Size of dataframe: {}".format(df.shape))

    classA, classB = seperateClass(df)

    print("\nSize of class A: {}".format(classA.shape))
    print("5 first rows in Class A:\n", classA.head())

    print("\nSize of class B: {}".format(classB.shape))
    print("5 first rows in Class B:\n", classB.head())

    # Show an example decision boundary graph (hand-picked coefficients)
    X1 = df['X1']  # total X1 from both class A and class B (plot on x axis)
    w0, w1 = 60, -0.32
    FX = w0 + w1 * X1  # fx is the decision boundary

    plot2Class(classA, classB, X1, FX)

    # Example: plot sigmoid value of class A and B
    # class A: X1 and X2
    X1A, X2A = splitFeature(classA)
    marginA = X2A - (w0 + w1 * X1A)
    s_A = sigmoid(marginA)
    plt.plot(marginA, s_A, 'ro')  # sigmoid is x axis

    # class B: X1 and X2
    X1B, X2B = splitFeature(classB)
    marginB = X2B - (w0 + w1 * X1B)
    s_B = sigmoid(marginB)
    plt.plot(marginB, s_B, 'bo')  # sigmoid is x axis
    plt.show()

    # show trend sigmoid
    margin = np.arange(-12, 70, 1)
    s = sigmoid(margin)
    plt.plot(marginA, s_A, 'ro')
    plt.plot(marginB, s_B, 'bo')
    plt.plot(margin, s, 'k-')
    plt.show()

    # formula
    # Z = w0 + w1*X1 + w2*X2
    # S = sigmoid(Z)

    # for easy normalization
    X1X2_mean = df[x_column_names].mean()
    dfNorm = df.copy()
    dfNorm[x_column_names] = dfNorm[x_column_names] / X1X2_mean

    train_X1X2 = dfNorm[x_column_names]                       # X1, X2 training set
    train_Y = dfNorm[y_column_name].values.reshape(-1, 1)     # Y (Output) training set

    C_List = train_method(train_X1X2, train_Y)

    classA, classB = seperateClass(dfNorm)
    X1A, X2A = splitFeature(classA)
    X1B, X2B = splitFeature(classB)

    # margin = w0 + w1*X1 + w2*X2
    # margin/w2 = w0/-w2 + (w1/-w2)*X1 - X2
    # ถ้าให้ 0 = w0/w2 + (w1/w2)*X1 - X2 มันคือสมการเส้นตรง ที่ทำนาย X2
    # w0/-w2 + (w1/-w2)*X1 -> the decision boundary Equation

    X1_norm = dfNorm['X1']
    X2_norm = dfNorm['X2']
    FX_List = getDecisionFunc(X1_norm, C_List)

    am.visualize(X1A, X2A, X1B, X2B, X1_norm, X2_norm, FX_List)
