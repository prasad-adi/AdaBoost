import pandas as pd
import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

np.random.seed(42)


def read_data(file_name):
    X_train = pd.read_csv("./spam_polluted/train_feature.txt", sep="\s", header=None)
    Y_train = pd.read_csv("./spam_polluted/train_label.txt", header=None)
    X_test = pd.read_csv("./spam_polluted/test_feature.txt", sep="\s", header=None)
    Y_test = pd.read_csv("./spam_polluted/test_label.txt", header=None)
    training_data = np.c_[X_train, Y_train]
    testing_data = np.c_[X_test, Y_test]
    np.random.shuffle(training_data)
    return training_data[:,:-1], training_data[:,-1], testing_data[:,:-1], testing_data[:,-1]

def normalise_data(data):
    r,c = data.shape
    for i in range(c - 1):
        temp = data[:, i]
        data[:, i] = (temp - np.mean(temp)) / np.std(temp)
    return data

def fit_gaussian_curves(X_train, Y_train):
    X_train_zero = X_train[np.where(Y_train == 0)]
    X_train_one = X_train[np.where(Y_train == 1)]
    mu0 = np.sum(X_train_zero, axis = 0) / (Y_train.shape[0] - np.sum(Y_train))
    mu1 = np.sum(X_train_one, axis = 0) / np.sum(Y_train)
    variance0 = X_train_zero - mu0
    sigma0 = (np.dot(variance0.T, variance0)) / (Y_train.shape[0] - np.sum(Y_train))
    variance1 = X_train_one - mu1
    sigma1 = (np.dot(variance1.T, variance1)) / np.sum(Y_train)
    np.fill_diagonal(sigma0, sigma0.diagonal() + 0.46)
    np.fill_diagonal(sigma1, sigma1.diagonal() + 0.46)
    identity_0 = np.identity(sigma0.shape[0])
    identity_1 = np.identity(sigma1.shape[0])
    np.fill_diagonal(identity_0, sigma0.diagonal())
    np.fill_diagonal(identity_1, sigma1.diagonal())
    return mu0, mu1, identity_0, identity_1

def get_predictions(X, Y, Y_train, mu0, mu1, sigma0, sigma1):
    #denominatorpi = np.power((2 * np.pi), (X.shape[1] / 2))
    denominatorpi = (X.shape[1] / 2) * np.log2(2 * np.pi)
    denominator0 = - denominatorpi + ((0.5) * np.log(np.linalg.det(sigma0)))
    #denominator0 = np.log2(1  / (np.sqrt(np.linalg.det(sigma0)) * denominatorpi))
    denominator1 = - denominatorpi + ((0.5) * np.log(np.linalg.det(sigma1)))
    #denominator1 = np.log2(1  / (np.sqrt(np.linalg.det(sigma1)) * denominatorpi))
    sigmaInverse0 = np.linalg.pinv(sigma0)
    sigmaInverse1 = np.linalg.pinv(sigma1)
    py0 = (Y_train.shape[0] - np.sum(Y_train)) / Y_train.shape[0]
    py1 = (np.sum(Y_train)) / Y_train.shape[0]
    y_pred = []
    probability_Y1 = []
    probability_Y0 = []
    for i in range(X.shape[0]):
        temp = X[i]
        variance0 = (temp - mu0)
        pxy0 = denominator0 * np.dot(np.dot(variance0.T, sigmaInverse0), variance0)
        pyx0 = ((pxy0 * py0) + 0.1) / X.shape[1]
        variance1 = temp - mu1
        pxy1 = denominator1 * np.dot(np.dot(variance1.T, sigmaInverse1), variance1)
        pyx1 = ((pxy1 * py1) + 0.1) / X.shape[1]
        probability_Y0.append(pyx1)
        probability_Y1.append(pyx0)
        if(pyx0 > pyx1):
            y_pred.append(0)
        else:
            y_pred.append(1)
    y_pred = np.array(y_pred)
    accuracy = sklearn.metrics.accuracy_score(Y, y_pred)
    plot_ROC_curve(np.array(probability_Y0), np.array(probability_Y1), Y)
    #print("accuracy = ", accuracy)
    return accuracy

def plot_ROC_curve(probability_Y0, probability_Y1, Y):
    log_probabilities = np.log2(probability_Y1 / probability_Y0)
    sorted_indices = np.argsort(log_probabilities)
    log_probabilities.sort()
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(Y[sorted_indices], log_probabilities)
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    print("AUC = ", roc_auc)



def k_split(X_train, Y_train, X_test, Y_test):
    mu0, mu1, sigma0, sigma1 = fit_gaussian_curves(X_train, Y_train)
    training_accuracy = get_predictions(X_train, Y_train, Y_train, mu0, mu1, sigma0, sigma1)
    test_accuracy = get_predictions(X_test, Y_test, Y_train, mu0, mu1,sigma0,sigma1)
    print("training_accuracy  = ", training_accuracy)
    print("test_accuracy = ", test_accuracy)

def get_PCA(training_data, testing_data):
    pca_reduction = PCA(n_components=100)
    train_size = training_data.shape[0]
    data = np.concatenate((training_data,testing_data), axis = 0)
    pca_data = pca_reduction.fit_transform(data)
    return pca_data[:train_size,:], pca_data[train_size:,:]


def run(pca):
    X_train, Y_train, X_test, Y_test = read_data("/Users/adityaprasad/Desktop/spambaseData/spambase.data.txt")
    if(pca):
        X_train, X_test = get_PCA(X_train, X_test)
    k_split(X_train, Y_train, X_test, Y_test)


run(pca = True)


