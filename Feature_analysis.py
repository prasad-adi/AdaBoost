import pandas as pd
import numpy as np

np.random.seed(0)
from AdaBoost_6_1 import AdaBoost_6_1

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
        boosting = AdaBoost_6_1()
        train_size = int(data.shape[0] * 0.8)
        training_data = data[:train_size,:]
        testing_data = data[train_size:,:]
        feature_alpha = boosting.boost(training_data,testing_data,  epochs)
        fraction = {}
        features = list(feature_alpha.keys())
        alphas = list(feature_alpha.values())
        total_sum_alphas = np.array(sum(alphas,[])).sum()
        for i in range(len(features)):
            alpha_f = np.abs(np.array(feature_alpha[features[i]]))
            fraction[features[i]] = (np.sum(alpha_f)) / total_sum_alphas
        predicted = np.array(sorted(fraction.items(), key=lambda x:-x[1])[:15]).T[0]
        answers = [52, 51, 56, 15, 6, 22, 23, 4, 26, 24, 7, 54, 5, 19, 18]
        count = 0
        for i in range(len(answers)):
            if(answers[i] not in predicted):
                count += 1
        print(count)


boost = Preprocess()
boost.run(normalise = False,epochs = 300)






