import pandas as pd
import numpy as np
from GradientBoosting import Gradient_Boosting
from GradientBoostinBuiltIn import Gradient_Boosting_BuiltIn

np.random.seed(0)

class Preprocess:
    def  __init__(self):
        pass

    def read_housing_data(self):
        dataset_train = pd.read_csv('housing_train.txt', sep='\s+', header=None)
        train_data = dataset_train.values
        X_train = dataset_train.iloc[:, :-1].values
        Y_train = dataset_train.iloc[:, 13].values

        dataset_test = pd.read_csv('housing_test.txt', sep='\s+', header=None)
        test_data = dataset_test.values
        X_test = dataset_test.iloc[:, :-1].values
        Y_test = dataset_test.iloc[:, 13].values
        return X_train, Y_train, X_test, Y_test


    def normalise_data(self, train_data, test_data):
        for i in range(train_data.shape[1]):
            temp = train_data[:, i]
            temp1 = test_data[:, i]
            mean = np.mean(temp)
            std = np.std(temp)
            train_data[:, i] = (temp - mean) / std
            test_data[:, i] = (temp1 - mean) / std
        return train_data, test_data



    def run(self, normalise, epochs, builtin):
        X_train, Y_train, X_test, Y_test = self.read_housing_data()
        if(normalise):
            X_train, X_test = self.normalise_data(X_train, X_test)
        if(builtin):
            boosting = Gradient_Boosting_BuiltIn()
        else:
            boosting = Gradient_Boosting()
        boosting.boost(X_train, Y_train, X_test, Y_test, epochs)

boost = Preprocess()
boost.run(normalise = True,epochs = 20, builtin=False)






