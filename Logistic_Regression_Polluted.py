#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 18:46:17 2019

@author: adityaprasad
"""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 14:52:10 2019

@author: adityaprasad
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics
from sklearn.decomposition import PCA
np.random.seed(0)


def normalise_data(X):
    for i in range(X.shape[1]):
        temp = X[:, i]
        X[:, i] = (temp - np.mean(temp)) / np.std(temp)
    return X

def linear_regression_Gradient_Descent(X, Y):
    r, c = X.shape
    np.random.seed(42)
    W = np.random.randn(c)
    lamb = 0.01
    for k in range(100):
        #if (k > 50):
             #lamb = 0.001
        # if (k > 70):
        #     lamb = 0.0001
        # if (k > 80):
        #     lamb = 0.00001
        # if (k > 90):
        #     lamb = 0.000001
        for i in range(c):
            H = np.matmul(X, W)
            G = 1 / (1 + np.exp(-H))
            diff = G - Y
            W = W - ((X[i] * (lamb * diff[i])) + (0.1 / X.shape[0])  * W)
    return W


def plot_roc_curve(y, y_test):
    p = np.linspace(min(y), max(y), 100)
    tpr = []
    fpr = []
    for j in p:
        y1 = []
        for i in range(len(y)):
            if (y[i] > j):
                y1.append(1)
            else:
                y1.append(0)
        y2 = np.array(y1)
        tp, fp, fn, tn = build_confusion_matrix(y2, y_test)
        t = tp / (tp + fn)
        f = fp / (fp + tn)
        tpr.append(t)
        fpr.append(f)
    plt.plot(fpr, tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    print("auc = ", -np.trapz(tpr, fpr))


def build_confusion_matrix(Y, y_test):
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    for i in range(len(Y)):
        if (y_test[i] == 1):
            if (Y[i] == 1):
                true_positive += 1
            if (Y[i] == 0):
                false_negative += 1
        if (y_test[i] == 0):
            if (Y[i] == 1):
                false_positive += 1
            if (Y[i] == 0):
                true_negative += 1
    return true_positive, false_positive, false_negative, true_negative


def fit(X_train, Y_train, X_test, Y_test):
    y_pred = linear_regression_Gradient_Descent(X_train, Y_train)
    y = np.matmul(X_test, y_pred)
    plot_roc_curve(y, Y_test)
    y1 = []
    for i in range(len(y)):
        if (y[i] > (0.5)):
            y1.append(1)
        else:
            y1.append(0)
    y2 = np.array(y1)
    true_positive, false_positive, false_negative, true_negative = build_confusion_matrix(y2, Y_test)
    print(true_positive, "\t", false_positive, "\n", false_negative, "\t", true_negative, "\n")
    build_confusion_matrix(y2, Y_test)
    error = np.sum(np.square(y2 - Y_test)) / len(Y_test)
    print("error = ", error)
    accuracy = sklearn.metrics.accuracy_score(Y_test, y2)
    print("accuracy = ", accuracy)

def read_data():
    X_train = pd.read_csv("./spam_polluted/train_feature.txt", sep="\s", header=None)
    Y_train = pd.read_csv("./spam_polluted/train_label.txt", header=None)
    X_test = pd.read_csv("./spam_polluted/test_feature.txt", sep="\s", header=None)
    Y_test = pd.read_csv("./spam_polluted/test_label.txt", header=None)
    training_data = np.c_[X_train, Y_train]
    testing_data = np.c_[X_test, Y_test]
    np.random.shuffle(training_data)

    return training_data[:, :-1], training_data[:, -1], testing_data[:, :-1], testing_data[:, -1]

def add_ones(X_train, X_test):
    X_train = np.c_[np.ones(X_train.shape[0]), X_train]
    X_test = np.c_[np.ones(X_test.shape[0]), X_test]
    return X_train, X_test

def get_PCA(training_data, testing_data):
    pca_reduction = PCA(n_components=100)
    train_size = training_data.shape[0]
    data = np.concatenate((training_data,testing_data), axis = 0)
    pca_data = pca_reduction.fit_transform(data)
    return pca_data[:train_size,:], pca_data[train_size:,:]


def run(normalise, pca):
    X_train, Y_train, X_test, Y_test = read_data()
    if(normalise):
        X_train = normalise_data(X_train)
        X_test = normalise_data(X_test)
    if(pca):
        X_train, X_test = get_PCA(X_train, X_test)

    X_train, X_test = add_ones(X_train, X_test)
    fit(X_train, Y_train, X_test, Y_test)

run(normalise=True, pca = False)








