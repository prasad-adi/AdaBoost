import pandas as pd
import numpy as np

np.random.seed(0)
from AdaBoostRandom import AdaBoostRandom
from AdaBoost_decision_stump import AdaBoost_decison_stump

class Preprocess:
    def  __init__(self):
        pass


    def _read_data_crx(self, filename):
        dataset_train = pd.read_csv(filename, sep = '\s', header=None)
        dataset_train = dataset_train.values
        dataset_train = np.where(dataset_train == '?', 0.0, dataset_train)
        dataset_train = pd.DataFrame(dataset_train)
        temp_dataset = pd.get_dummies(dataset_train.iloc[:,:-1], columns=[0,3,4,5,6,8,9,11,12], dtype=float)

        dataset = np.c_[temp_dataset.values, dataset_train.iloc[:,-1].values]
        dataset[:,-1] = np.where(dataset[:,-1] == "+", 1, -1)
        dataset = np.delete(dataset, [6, 9, 13, 17, 32,], axis=1)
        dataset = dataset.astype(float)
        np.random.shuffle(dataset)
        return dataset

    def _read_data_vote(self, filename):
        dataset_train = pd.read_csv(filename, sep = '\s', header = None)
        dataset_train = dataset_train.values
        dataset_train = np.where(dataset_train == '?', 0.0, dataset_train)
        dataset_train = pd.DataFrame(dataset_train)
        temp_dataset = pd.get_dummies(dataset_train.iloc[:, :-1], dtype=float)
        dataset = np.c_[temp_dataset.values, dataset_train.iloc[:, -1].values]
        dataset[:, -1] = np.where(dataset[:, -1] == "d", 1, -1)
        dataset = np.delete(dataset, [0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45], axis = 1)
        dataset = dataset.astype(float)
        np.random.shuffle(dataset)
        return dataset

    def _read_data_spam(self, filename):
        dataset_train = pd.read_csv(filename, header=None)
        dataset = dataset_train.values
        np.random.shuffle(dataset)
        return dataset


    def normalise_data(self, dataset):
        for i in range(dataset.shape[1] - 1):
            temp = dataset[:, i]
            dataset[:, i] = (temp - np.mean(temp)) / np.std(temp)
        return dataset



    def run(self, normalise, epochs, random_classifier, k_split, database, bagging):
        filenames = ["spambase.data.txt", "crx.data", "vote.data"]
        if(database == 1):
            data = self._read_data_crx(filenames[database])
        elif(database == 0):
            data = self._read_data_spam(filenames[database])
            indices = np.where(data[:, -1] == 0)
            data[:, -1][indices] = -1
        elif(database == 2):
            data = self._read_data_vote(filenames[database])
        if(normalise):
            data = self.normalise_data(data)

        n = np.array_split(data, k_split)
        training_accuracy = 0
        testing__accuracy = 0
        for i in range(k_split):
            data_temp = n[i]
            X_test = data_temp[:, :-1]
            Y_test = data_temp[:, -1]
            counter = 0
            for j in range(k_split):
                if (j != i):
                    if (counter == 0):
                        data_temp = n[j]
                        counter += 1
                    else:
                        data_temp = np.concatenate((data, n[j]), axis=0)
            X_train = data_temp[:, :-1]
            Y_train = data_temp[:, -1]
            if (random_classifier):
                boosting = AdaBoostRandom()
            else:
                boosting = AdaBoost_decison_stump()
            training_accuracy_temp, testing_accuracy_temp = boosting.boost(data, np.c_[X_test,  Y_test],  epochs)
            training_accuracy += training_accuracy_temp
            testing__accuracy += testing_accuracy_temp
        print("Average_training_accuracy = ", training_accuracy / k_split)
        print("Average testing accuracy = ", testing__accuracy / k_split)
        c = []

boost = Preprocess()
boost.run(normalise = False,epochs = 201, random_classifier= False, k_split= 10, database = 1, bagging = True)






