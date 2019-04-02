import numpy as np
from weak_classifier_bagging import weak_classifier_bagging
import collections
from sklearn.metrics import mean_squared_error
#from bagging_classifier import weak_classifier_decision_stump_2
import sklearn


class Bagging:
    def fit(self, training_data, testing_data, epochs):
        sample_size = 200
        training_predictions = []
        testing_predictions = []
        for i in range(epochs):
            print(i)
            indices = np.random.randint(0, training_data.shape[0], size=sample_size)
            classifier = weak_classifier_bagging()
            tree  = classifier.decisionTree(training_data[indices], 10, 1, 0)
            temp_train_predictions = classifier.predict(tree, training_data)
            temp_test_predictions = classifier.predict(tree, testing_data)
            print("training accuracy for ",i, " = ",sklearn.metrics.accuracy_score(training_data[:,-1], temp_train_predictions))
            print("testing accuracy for ", i, " = ",sklearn.metrics.accuracy_score(testing_data[:, -1], temp_test_predictions))
            training_predictions.append(temp_train_predictions)
            testing_predictions.append(temp_test_predictions)
        y_pred_train = self.__get_most_common(np.array(training_predictions))
        y_pred_test = self.__get_most_common(np.array(testing_predictions))
        print("training_error = ", sklearn.metrics.accuracy_score(training_data[:,-1], y_pred_train))
        print("testing_error = ", sklearn.metrics.accuracy_score(testing_data[:,-1], y_pred_test))

    def __get_most_common(self, predictions):
        y_pred = []
        for i in range(predictions.shape[1]):
            y_pred.append(collections.Counter(predictions.T[i]).most_common(1)[0][0])
        return y_pred




