import pandas as pd
import numpy as np
import ray
import resource
import sys
import os
from scipy.spatial import distance
from AdaBoost_ECOC import AdaBoost_ECOC
import sklearn

ray.init()

@ray.remote
def read_and_process_data(filename, train_flag):
    with open(filename, "r") as f:
        contents = f.read()
        contents = contents.split("\n")
        X = np.zeros(1754).reshape(1, 1754)
        Y = []
        for i in range(len(contents) - 1):
            if(i % 1000 == 0):
                print(i)
            temp_array = np.zeros(1754)

            temp = contents[i].split(" ")
            Y.append(int(temp[0]))
            for j in range(1, len(temp) - 1):
                index_value = temp[j].split(":")
                temp_array[int(index_value[0])] = float(index_value[1])
            X = np.concatenate((X,temp_array.reshape(1, 1754)), axis = 0)
    if(train_flag):
        np.savetxt("./X_train.txt", X)
        np.savetxt("./Y_train.txt", X = np.array(Y))
    else:
        np.savetxt("./X_test.txt", X)
        np.savetxt("./Y_test.txt", X = np.array(Y))
    return X, np.array(Y)

def get_label(column, class_codes, Y):
    labels = class_codes.T[column]
    temp_labels = []
    for i in range(Y.shape[0]):
        #print(i)
        temp_labels.append(int(labels[int(Y[i])]))
    return np.array(temp_labels).reshape(Y.shape[0], 1)


def get_random_codes(class_codes, ecoc_number):
    list = np.random.randint(0, class_codes.shape[1], size=ecoc_number)
    return list

def get_class_from_ECOC(testing_predictions, class_codes):
    label = []
    for i in range(testing_predictions.shape[0]):
        hamming_distances = []
        for j in range(class_codes.shape[0]):
            hamming_distances.append(distance.hamming(testing_predictions[i], class_codes[j]))
        label.append(np.array(hamming_distances).argmin())
    return label

@ray.remote
def call_boost(list, class_codes, i,X_train, Y_train, X_test, Y_test ):
    boosting = AdaBoost_ECOC()
    Y = get_label(list[i], class_codes, Y_train)
    testing_predictions, training_predictions = boosting.boost(np.c_[X_train, Y], np.c_[X_test, Y_test], 200)
    testing_predictions = np.where(testing_predictions <= 0, -1, 1)
    training_predictions = np.where(training_predictions <= 0, -1, 1)
    return testing_predictions, training_predictions



def  run(class_codes, X_train, Y_train, X_test, Y_test, ecoc_number):
    list = get_random_codes(class_codes, ecoc_number)
    testing_predictions_ecoc = np.zeros(X_test.shape[0]).reshape(X_test.shape[0], 1)
    training_predictions_ecoc = np.zeros(X_train.shape[0]).reshape(X_train.shape[0], 1)
    i = 0
    while(i < ecoc_number):
        print(i)
        Id1 = call_boost.remote(list, class_codes, i, X_train, Y_train, X_test, Y_test)
        Id2 = call_boost.remote(list, class_codes, i + 1, X_train, Y_train, X_test, Y_test)
        Id3 = call_boost.remote(list,  class_codes, i + 2, X_train, Y_train, X_test, Y_test)
        testing_predictions0, training_predictions0 = ray.get(Id1)
        testing_predictions1, training_predictions1 = ray.get(Id2)
        testing_predictions2, training_predictions2 = ray.get(Id3)
        testing_predictions_ecoc = np.concatenate((testing_predictions_ecoc,
                                              testing_predictions0.reshape(X_test.shape[0], 1)), axis = 1)
        testing_predictions_ecoc = np.concatenate((testing_predictions_ecoc,
                                                   testing_predictions1.reshape(X_test.shape[0], 1)), axis=1)
        testing_predictions_ecoc = np.concatenate((testing_predictions_ecoc,
                                                   testing_predictions2.reshape(X_test.shape[0], 1)), axis=1)

        training_predictions_ecoc = np.concatenate((training_predictions_ecoc, training_predictions0.reshape(X_train.shape[0], 1)), axis = 1)
        training_predictions_ecoc = np.concatenate(
            (training_predictions_ecoc, training_predictions1.reshape(X_train.shape[0], 1)), axis=1)
        training_predictions_ecoc = np.concatenate(
            (training_predictions_ecoc, training_predictions1.reshape(X_train.shape[0], 1)), axis=1)
        if (i % 6 == 0):
            y_pred_test = get_class_from_ECOC(testing_predictions_ecoc[:, 1:], class_codes[:, list])
            y_pred_train = get_class_from_ECOC(training_predictions_ecoc[:, 1:], class_codes[:, list])
            print("ECOC training accuracy = ", sklearn.metrics.accuracy_score(Y_train, y_pred_train))
            print("ECOC testing accuracy = ", sklearn.metrics.accuracy_score(Y_test, y_pred_test))
        i += 3

def generate_class_codes(k):
    class_codes = np.ones(127).reshape(1,127)
    for i in range(1, k):
        temp = np.zeros(1).reshape(1,1)
        for j in range(0, int((2 ** i) / 2)):
            if(j == int((2 ** i) / 2)  - 1):
                temp = np.concatenate((temp, np.zeros(np.power(2, k - (i + 1))).reshape(1, np.power(2, k - (i + 1)))), axis = 1)
                temp = np.concatenate((temp, np.ones(np.power(2, k - (i+1)) - 1).reshape(1, np.power(2, k - (i+1)) - 1 )), axis = 1)

            else:
                temp = np.concatenate((temp, np.zeros(np.power(2, k - (i + 1))).reshape(1, np.power(2, k - (i + 1)))), axis = 1)
                temp = np.concatenate((temp, np.ones(np.power(2, k - (i + 1))).reshape(1, np.power(2, k - (i + 1)))), axis = 1)
        class_codes = np.concatenate((class_codes, temp[:,1:].reshape(1, 127)), axis = 0)
    return class_codes

@ray.remote
def read(filename):
    data = np.loadtxt(filename)
    return data

X_train, Y_train = read_and_process_data("./8newsgroup/train.trec/feature_matrix.txt", 1)
print("Read X_train")
X_test, Y_test = read_and_process_data("./8newsgroup/test.trec/feature_matrix.txt", 0)
#X_train_Id = read.remote("./X_train.txt")
#Y_train_Id = read.remote("./Y_train.txt")
print("Read  Y_train")
#X_test_Id = read.remote("./X_test.txt")
print("Read X_test")
#Y_test_Id = read.remote("./Y_test.txt")
#X_train = ray.get(X_train_Id)
#Y_train = ray.get(Y_train_Id)
#X_test = ray.get(X_test_Id)
#Y_test = ray.get(Y_test_Id)
print("Read Y_test")
X_train = X_train[1:,:]
X_test = X_test[1:,:]
class_codes = generate_class_codes(8)
class_codes = np.where(class_codes == 0, -1, class_codes)
run(class_codes, X_train, Y_train.reshape(Y_train.shape[0],1), X_test, Y_test.reshape(Y_test.shape[0], 1), 30)
