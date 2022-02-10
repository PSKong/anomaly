# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 15:46:06 2018
@author: admin
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
from sklearn import datasets
import scipy.io as sio
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from functions import make_data, preK


def loadDataSet(x, y):
    # dataMat = np.mat(x)
    y = np.mat(y)
    b = np.ones(y.shape)
    X = np.column_stack((b, x))
    X = np.mat(X)
    labeltype = np.unique(y.tolist())
    eyes = np.eye(len(labeltype))
    Y = np.zeros((X.shape[0], len(labeltype)))
    for i in range(X.shape[0]):
        Y[i, :] = eyes[int(y[i, 0])]
    return X, y, Y



def data_convert(x, y):
    b = np.ones(y.shape)
    x_b = np.column_stack((b, x))
    K = len(np.unique(y.tolist()))
    eyes_mat = np.eye(K)
    y_onehot = np.zeros((y.shape[0], K))
    for i in range(0, y.shape[0]):
        y_onehot[i] = eyes_mat[y[i]]
    return x_b, y, y_onehot


def softmax(s):
    return np.exp(s) / np.sum(np.exp(s), axis=1)





def gradAscent(x, y, alpha=0.05, max_loop=500):


    weights = np.ones((x.shape[1], y.shape[1]))
    print('weights初始化值：', weights)
    for k in range(max_loop):
        # k=0
        h = softmax(x * weights)
        error = h - y
        weights = weights - alpha * x.T * error
    return weights


def SoftmaxGD(x, y, alpha=0.05, max_loop=500):

    m = np.shape(x)[1]
    n = np.shape(y)[1]
    weights = np.ones((m, n))

    for k in range(max_loop):
        # k=2
        h = softmax(x * weights)
        P = h
        error = y - h
        weights = weights + alpha * x.transpose() * error  # 梯度下降算法公式
    # print('k:',k,'weights:',weights.T)
    return weights.getA(), P


def SoftmaxSGD(x, y, X_, y_, alpha=0.05, max_loop=2000):

    m = np.shape(x)[1]
    n = np.shape(y)[1]
    p = np.shape(x)[0]
    weights = np.ones((m, n))
    P = np.ones((p, n))

    for k in range(max_loop):
        for i in range(0, len(x)):
            # k=0;i=0
            h = softmax(x[i] * weights)
            P[i, :] = h
            error = y[i] - h[0]
            weights = weights + alpha * x[i].T * error[0]


        y_hat2_ = predict(weights, X_)
        CM = confusion_matrix(y_.A, y_hat2_)
        ACC_CM_0 = CM[0, 0] / (CM[0, 0] + CM[0, 1])
        ACC_CM_1 = CM[1, 1] / (CM[1, 0] + CM[1, 1])

        print("Epoch:{}, ACC_CM_0:{:.4f}, ACC_CM_1:{:.4f}".format(k+1, ACC_CM_0, ACC_CM_1))
    return weights.getA(), P


def Softmax(x, y, alpha=0.05, max_loop=50):

    n = np.shape(y)[1]
    p = np.shape(x)[0]
    P = np.ones((p, n))

    P = softmax(x)

    return P


def plotBestFit(dataMat, labelMat, weights):

    x1_min, x1_max = dataMat[:, 1].min() - .5, dataMat[:, 1].max() + .5
    x2_min, x2_max = dataMat[:, 2].min() - .5, dataMat[:, 2].max() + .5

    step = 0.02
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, step), np.arange(x2_min, x2_max, step))
    testMat = np.c_[xx1.ravel(), xx2.ravel()]
    testMat = np.column_stack((np.ones(((testMat.shape[0]), 1)), testMat))
    testMat = np.mat(testMat)
    y = softmax(testMat * weights)

    predicted = y.argmax(axis=1)
    predicted = predicted.reshape(xx1.shape).getA()

    plt.pcolormesh(xx1, xx2, predicted, cmap=plt.cm.Paired)

    plt.scatter(dataMat[:, 1].flatten().A[0], dataMat[:, 2].flatten().A[0],
                c=labelMat.flatten().A[0], alpha=.5)
    plt.show()


def predict(weights, testdata):
    y_hat = softmax(testdata * weights)
    predicted = y_hat.argmax(axis=1).getA()
    return predicted


if __name__ == "__main__":

    train_data, train_lab, test_data, test_lab = make_data()



    x = train_data
    y = train_lab

    x = np.mat(x)
    y = np.mat(y).T


    X, Y, Y_onehot = data_convert(x, y)

    x_ = test_data
    y_ = test_lab
    x_ = np.mat(x_)
    y_ = np.mat(y_).T
    X_, Y_, Y_onehot_ = data_convert(x_, y_)

    weights2, P = SoftmaxSGD(X, Y_onehot, X_, y_)

    y_hat = softmax(X_ * weights2)
    y_hat2 = predict(weights2, X_)

