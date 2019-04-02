import numpy as np
np.random.seed(42)

class weak_classifier_random:
    def __init__(self, data, distributions):
        self.data = data
        self.distributions = distributions
        self.labels = data[:, -1]
        self.max_error = 0
        self.max_split_column = 0
        self.max_split_value = 0

    def build_stump(self):
        thresholds = self.get_threshold()
        random_choice_column = np.random.randint(0, len(thresholds), size = 1)[0]
        random_choice_value = np.random.randint(0,len(thresholds[random_choice_column]), size = 1)[0]
        self.max_split_value = thresholds[random_choice_column][random_choice_value]
        self.max_split_column = random_choice_column
        return self.max_split_column, self.max_split_value

    def get_threshold(self):
        thresholds = []
        for i in range(self.data.shape[1] - 1):
            temp_threshold = []
            temp = self.data[:, i]
            sorted = np.unique(temp)
            temp_threshold.append(sorted[0] - 1)
            for i in range(1, len(sorted)):
                temp_threshold.append((sorted[i] + sorted[i - 1]) / 2)
            temp_threshold.append(sorted[len(sorted) - 1] + 1)
            thresholds.append(temp_threshold)
        return thresholds

    def predict(self, data):
        labels = np.zeros(data.shape[0])
        labels[np.where(data[:,self.max_split_column] <= self.max_split_value)] = -1
        labels[np.where(data[:,self.max_split_column] > self.max_split_value)] = 1
        return labels
