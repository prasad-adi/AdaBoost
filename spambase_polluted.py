import pandas as pd
import numpy as np

np.random.seed(0)
from AdaBoostRandom import AdaBoostRandom
from AdaBoost_decision_stump import AdaBoost_decison_stump

class Preprocess:
    def  __init__(self):
        pass

    def _read_data_spam(self):
        X_train = pd.read_csv("./spam_polluted/train_feature.txt", sep="\s", header=None)
        Y_train = pd.read_csv("./spam_polluted/train_label.txt", header=None)
        X_test = pd.read_csv("./spam_polluted/test_feature.txt", sep="\s", header=None)
        Y_test = pd.read_csv("./spam_polluted/test_label.txt", header=None)
        training_data = np.c_[X_train, Y_train]
        testing_data = np.c_[X_test,Y_test]
        np.random.shuffle(training_data)
        return training_data, testing_data


    def normalise_data(self, dataset):
        for i in range(dataset.shape[1] - 1):
            temp = dataset[:, i]
            dataset[:, i] = (temp - np.mean(temp)) / np.std(temp)
        return dataset



    def run(self, normalise, epochs, random_classifier):
        training_data, testing_data = self._read_data_spam()
        indices_training = np.where(training_data[:, -1] == 0)
        training_data[:, -1][indices_training] = -1
        indices_testing = np.where(testing_data[:,-1]==0)
        testing_data[:,-1][indices_testing] = -1
        print("read_data")
        if(normalise):
            training_data = self.normalise_data(training_data)
            testing_data = self.normalise_data(testing_data)
        if (random_classifier):
            boosting = AdaBoostRandom()
        else:
            boosting = AdaBoost_decison_stump()
        training_accuracy, testing_accuracy = boosting.boost(training_data, testing_data,  epochs)
        c = []

        print("Average_training_accuracy = ", training_accuracy)
        #print("Average testing accuracy = ", testing__accuracy)
        c = []

boost = Preprocess()
boost.run(normalise = False,epochs = 201, random_classifier= False)






