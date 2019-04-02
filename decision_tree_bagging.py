import pandas as pd
import numpy as np

np.random.seed(0)
#from Bagging import Bagging
from Bagging_decision_tree import Bagging

class Preprocess:
    def  __init__(self):
        pass

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

    def run(self, normalise, epochs):
        data = self._read_data_spam("spambase.data.txt")
        indices = np.where(data[:, -1] == 0)
        data[:, -1][indices] = -1
        if(normalise):
            data = self.normalise_data(data)
        training_size = int(data.shape[0] * 0.8)
        training_data = data[:training_size,:]
        testing_data = data[training_size:,:]
        bagging = Bagging()
        bagging.fit(training_data, testing_data, epochs)




boost = Preprocess()
boost.run(normalise = True,epochs = 50)
