import numpy as np
np.random.seed(0)

import sklearn
from scipy.spatial import distance
from Ada_Boost_Ecoc_Random import AdaBoostRandom

def read_data(filename, train):
    with open(filename, "r") as f:
        if(train):
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

def get_class_from_ECOC(testing_predictions, class_codes, ecoc_number):
    label = []
    for i in range(testing_predictions.shape[0]):
        hamming_distances = []
        for j in range(class_codes.shape[0]):
            hamming_distances.append(distance.hamming(testing_predictions[i], class_codes[j]))
        label.append(np.array(hamming_distances).argmin())
    return label

def  run(class_codes, X_train, Y_train, X_test, Y_test, ecoc_number):
    list = get_random_codes(class_codes, ecoc_number)
    testing_predictions_ecoc = np.zeros(X_test.shape[0]).reshape(X_test.shape[0], 1)
    training_predictions_ecoc = np.zeros(X_train.shape[0]).reshape(X_train.shape[0], 1)
    for i in range(ecoc_number):
        boosting = AdaBoostRandom()
        Y = get_label(list[i], class_codes, Y_train)
        testing_predictions, training_predictions = boosting.boost(np.c_[X_train, Y], np.c_[X_test, Y_test], 1000)
        testing_predictions = np.where(testing_predictions <= 0, -1, 1)
        training_predictions = np.where(training_predictions <= 0, -1, 1)
        testing_predictions_ecoc = np.concatenate((testing_predictions_ecoc,
                                              testing_predictions.reshape(X_test.shape[0], 1)), axis = 1)
        training_predictions_ecoc = np.concatenate((training_predictions_ecoc, training_predictions.reshape(X_train.shape[0], 1)), axis = 1)
        y_pred_test = get_class_from_ECOC(testing_predictions_ecoc[:,1:], class_codes[:,list[:i+1]], ecoc_number)
        y_pred_train = get_class_from_ECOC(training_predictions_ecoc[:,1:], class_codes[:, list[:i+1]], ecoc_number)
        print("ECOC training accuracy = ", sklearn.metrics.accuracy_score(Y_train, y_pred_train))
        print("ECOC testing accuracy = ", sklearn.metrics.accuracy_score(Y_test, y_pred_test))




X_train, Y_train = read_data("./8newsgroup/train.trec/feature_matrix.txt", 1)
print("Read X_train")
X_test, Y_test = read_data("./8newsgroup/test.trec/feature_matrix.txt", 0)
#X_train = np.loadtxt("./X_train.txt")
#Y_train = np.loadtxt("./Y_train.txt")
print("Read  Y_train")
#X_test = np.loadtxt("./X_test.txt")
print("Read X_test")
#Y_test = np.loadtxt("./Y_test.txt")
print("Read Y_test")
#X_train = X_train[1:,:]
#X_test = X_test[1:,:]
class_codes = generate_class_codes(8)
class_codes = np.where(class_codes == 0, -1, class_codes)
print("finished_reading_data")
c = []
#list = get_random_codes(class_codes, 20)
#Y = get_label(list[0], class_codes, Y_train.reshape(Y_train.shape[0],1))
run(class_codes, X_train, Y_train.reshape(Y_train.shape[0],1), X_test, Y_test.reshape(Y_test.shape[0], 1), 50)



