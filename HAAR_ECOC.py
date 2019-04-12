import pandas as pd
import numpy as np
import ray
from scipy.spatial import distance
from AdaBoost_ECOC import AdaBoost_ECOC
import sklearn
ray.init()
from mnist2image.mnist import load_mnist
from Extract_HAAR import HaarFeatures

def read_data(filename, train):
    with open(filename, "r") as f:
        if (train):
            matrix = np.zeros((11314, 1755))
        else:
            matrix = np.zeros((7532, 1755))
        i = 0
        for line in f:
            temp_line = line.strip()
            dataset = temp_line.split()
            matrix[i][1754] = dataset[0]
            for data in dataset[1:]:
                column, value = data.split(":")
                matrix[i][int(column)] = value
            i +=1
    return matrix[:,:-1], matrix[:,-1]


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
    return np.array(label)

@ray.remote
def call_boost(list, class_codes, i,X_train, Y_train, X_test, Y_test):
    boosting = AdaBoost_ECOC()
    Y = get_label(list[i], class_codes, Y_train)
    testing_predictions, training_predictions = boosting.boost(np.c_[X_train, Y], np.c_[X_test, Y_test], 200)
    testing_predictions = np.where(testing_predictions <= 0, -1, 1)
    training_predictions = np.where(training_predictions <= 0, -1, 1)
    return testing_predictions, training_predictions

def concatenate(ecoc, predictions0, predictions1, predictions2, X):
    ecoc = np.concatenate((ecoc, predictions0.reshape(X.shape[0], 1)), axis=1)
    ecoc = np.concatenate((ecoc, predictions1.reshape(X.shape[0], 1)), axis=1)
    ecoc = np.concatenate((ecoc, predictions2.reshape(X.shape[0], 1)), axis=1)
    return ecoc


def  run(class_codes, X_train, Y_train, X_test, Y_test, ecoc_number):
    list = get_random_codes(class_codes, ecoc_number)
    testing_predictions_ecoc = np.zeros(X_test.shape[0]).reshape(X_test.shape[0], 1)
    training_predictions_ecoc = np.zeros(X_train.shape[0]).reshape(X_train.shape[0], 1)
    i = 0
    with open("./HAAR.txt", "w") as f:
        while(i < ecoc_number):
            print(i)
            Id1 = call_boost.remote(list, class_codes, i, X_train, Y_train, X_test, Y_test)
            Id2 = call_boost.remote(list, class_codes, i + 1, X_train, Y_train, X_test, Y_test)
            Id3 = call_boost.remote(list,  class_codes, i + 2, X_train, Y_train, X_test, Y_test)
            testing_predictions0, training_predictions0 = ray.get(Id1)
            testing_predictions1, training_predictions1 = ray.get(Id2)
            testing_predictions2, training_predictions2 = ray.get(Id3)
            testing_predictions_ecoc = concatenate(testing_predictions_ecoc, testing_predictions0, testing_predictions1, testing_predictions2,
                                                   X_test)
            training_predictions_ecoc = concatenate(training_predictions_ecoc, training_predictions0, training_predictions1, training_predictions2,
                                                    X_train)
            i += 3
            print("ecoc number = ", i)
            y_pred_test = get_class_from_ECOC(testing_predictions_ecoc[:, 1:], class_codes[:, list[:i]])
            y_pred_train = get_class_from_ECOC(training_predictions_ecoc[:, 1:], class_codes[:, list[:i]])
            f.write("ECOC training accuracy = " + str(sklearn.metrics.accuracy_score(Y_train, y_pred_train)))
            f.write("ECOC testing accuracy = " +str(sklearn.metrics.accuracy_score(Y_test, y_pred_test)))
            print("ECOC training accuracy = ", sklearn.metrics.accuracy_score(Y_train, y_pred_train))
            print("ECOC testing accuracy = ", sklearn.metrics.accuracy_score(Y_test, y_pred_test))


def generate_class_codes(k):
    size = (2 ** (k-1) - 1)
    class_codes = np.ones(size).reshape(1,size)
    for i in range(1, k):
        temp = np.zeros(1).reshape(1,1)
        for j in range(0, int((2 ** i) / 2)):
            if(j == int((2 ** i) / 2)  - 1):
                temp = np.concatenate((temp, np.zeros(np.power(2, k - (i + 1))).reshape(1, np.power(2, k - (i + 1)))), axis = 1)
                temp = np.concatenate((temp, np.ones(np.power(2, k - (i+1)) - 1).reshape(1, np.power(2, k - (i+1)) - 1 )), axis = 1)

            else:
                temp = np.concatenate((temp, np.zeros(np.power(2, k - (i + 1))).reshape(1, np.power(2, k - (i + 1)))), axis = 1)
                temp = np.concatenate((temp, np.ones(np.power(2, k - (i + 1))).reshape(1, np.power(2, k - (i + 1)))), axis = 1)
        class_codes = np.concatenate((class_codes, temp[:,1:].reshape(1, size)), axis = 0)
    return class_codes

@ray.remote
def read(filename):
    data = np.loadtxt(filename)
    return data

def read_mnist():
    X, Y = load_mnist()
    main_indices = []
    for i in range(10):
        indices = np.where(Y == i)[0]
        size = int(len(indices) * 0.5)
        indices_20 = np.random.randint(0, len(indices), size=size)
        main_indices.extend(indices[indices_20])
    return X[main_indices], Y[main_indices]




import zipfile
with zipfile.ZipFile("./features.txt.zip","r") as zip_ref:
    zip_ref.extractall("./")
X = np.loadtxt("./features.txt")
Y = np.loadtxt("./labels.txt")
data = np.c_[X, Y]
np.random.shuffle(data)
train_size = int(data.shape[0] * 0.9)
X_train = data[:train_size, :-1]
X_test = data[train_size:,:-1]
Y_train = data[:train_size,-1]
Y_test = data[train_size: ,-1]

print("done reading")
class_codes = generate_class_codes(10)
print("got class codes")
class_codes = np.where(class_codes == 0, -1, class_codes)
run(class_codes, X_train, Y_train.reshape(Y_train.shape[0],1), X_test, Y_test.reshape(Y_test.shape[0], 1), 51)
