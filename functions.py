import scipy.io as sio
import numpy as np
import random as rd

def make_data():
    X = sio.loadmat('data/test/X_tf_.mat')
    x = X['X']
    Label = sio.loadmat('data/test/Label.mat')
    z = Label['Label']
    y = np.ones(z.shape[0])
    y = z[:, 0]

    a = 5180

    train_a_1 = 208
    test = 2
    train_a_0 = train_a_1 * 1

    x_tf_test = np.ones((test, 64))
    y_tf_test = np.ones((test)).astype(int)

    x_tf_test[0, :] = x[a, :]
    y_tf_test[0] = y[a]
    x_tf_test[1, :] = x[0, :]
    y_tf_test[1] = y[0]

    x = np.delete(x, a, axis=0)
    y = np.delete(y, a)


    x_tf_train = np.ones((train_a_0 + train_a_1, 64))
    y_tf_train = np.ones((train_a_0 + train_a_1)).astype(int)
    x_0 = np.ones((train_a_0, 64))
    y_0 = np.ones((train_a_0)).astype(int)
    x_1 = np.ones((train_a_1, 64))
    y_1 = np.ones((train_a_1)).astype(int)

    i_0 = 0
    i_1 = 0
    for i_2 in range(np.shape(x)[0]):
        if (i_0 < np.shape(x_0)[0] and y[i_2] == 0):
            x_0[i_0, :] = x[i_2, :]
            y_0[i_0] = y[i_2]
            i_0 = i_0 + 1
        if (i_1 < np.shape(x_1)[0] and y[i_2] == 1):
            x_1[i_1, :] = x[i_2, :]
            y_1[i_1] = y[i_2]
            i_1 = i_1 + 1

    x_tf_train[0:train_a_0, :] = x_0[0:train_a_0, :]
    y_tf_train[0:train_a_0] = y_0[0:train_a_0]
    x_tf_train[train_a_0:train_a_0+train_a_1, :] = x_1[0:train_a_1, :]
    y_tf_train[train_a_0:train_a_0+train_a_1] = y_1[0:train_a_1]


    return x_tf_train, y_tf_train, x_tf_test, y_tf_test


def preK(y_out, y):
    m = np.shape(y)[0]
    P_ = np.ones((m, 3))
    P_[:, 2] = y_out
    P_[:, 1] = y
    for i in range(m):
        P_[i, 0] = i

    a = P_
    b = a[np.lexsort(-a.T)]#倒序

    ACC = np.ones(4)
    i_2 = 0
    for i_1 in [50, 100, 200, 300]:
        j = 0
        for i in range(0, i_1):
            if b[i, 1].astype(int) == 1:
                j = j + 1

        ACC[i_2] = j / i_1
        i_2 = i_2 + 1

    return ACC