import pandas as pd
import numpy as np

np.random.seed(0)
from AdaBoostRandom import AdaBoostRandom
from AdaBoost_Active_Learning import AdaBoost_Active_Learning

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



    def run(self, normalise, epochs, random_classifier, database):
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
        train_size = int(data.shape[0] * (0.05))
        training_indices = np.random.randint(0, data.shape[0], size = (train_size))
        testing_indices = [j for j in range(data.shape[0]) if j not in training_indices]
        training_data = data[training_indices]
        testing_data = data[testing_indices]
        increment_size = int(data.shape[0] * 0.02)
        i = 0
        while(len(training_indices) <= int(data.shape[0] * 0.5)):
            if (random_classifier):
                boosting = AdaBoostRandom()
            else:
                boosting = AdaBoost_Active_Learning()
            training_accuracy, testing_accuracy, testing_prediction = boosting.boost(training_data, testing_data,  epochs)
            print("training accuracy at ",int(5 + i)," = ",   training_accuracy)
            print("testing_accuracy at ",int(5+i)," = ",  testing_accuracy)
            #sorted_indices = np.argsort(testing_prediction)
            temp_prediction = testing_prediction
            new_indices = []
            for k in range(increment_size):
                if(temp_prediction.size!=0):
                    index = np.abs(temp_prediction - 0.0).argmin()
                    new_indices.append(testing_indices[index])
                    temp_prediction = np.delete(temp_prediction, index)
            new_indices = np.array(new_indices)
            training_indices = np.concatenate((training_indices, new_indices))
            training_data = data[training_indices]
            testing_indices = np.setdiff1d(testing_indices, new_indices)
            testing_data = data[testing_indices]
            i = i + 2














boost = Preprocess()
boost.run(normalise = False,epochs = 101, random_classifier= False, database = 0)






