import numpy as np
#from weak_classifier_bagging import weak_classifier_bagging
from sklearn.metrics import mean_squared_error
from bagging_classifier import weak_classifier_decision_stump_2
import sklearn


class Bagging:
    def fit(self, training_data, testing_data, epochs):
        sample_size = int(training_data.shape[0] * 0.02)
        training_predictions = []
        testing_predictions = []
        for i in range(epochs):
            indices = np.random.randint(0, training_data.shape[0], size=sample_size)
            classifier = weak_classifier_decision_stump_2()
            error, column, value  = classifier.fit(training_data[indices][:,:-1], training_data[indices][:,-1])
            temp_train_predictions = classifier.predict(column = column, value = value, X = training_data)
            temp_test_predictions = classifier.predict(column, value, testing_data)
            print("training accuracy for ",i, " = ",sklearn.metrics.accuracy_score(training_data[:,-1], temp_train_predictions))
            print("testing accuracy for ", i, " = ",sklearn.metrics.accuracy_score(testing_data[:, -1], temp_test_predictions))
            training_predictions.append(temp_train_predictions)
            testing_predictions.append(temp_test_predictions)
        y_pred_train = np.array(training_predictions).mean(axis = 0)
        y_pred_test = np.array(testing_predictions).mean(axis = 0)
        print("training_error = ", mean_squared_error(training_data[:,-1], y_pred_train))
        print("testing_error = ", mean_squared_error(testing_data[:,-1], y_pred_test))




