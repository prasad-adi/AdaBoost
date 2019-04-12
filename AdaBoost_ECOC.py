import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt
np.random.seed(0)
from weak_classifier_decision_Stump import weak_classifier_decision_stump_2


class AdaBoost_ECOC:
    def __init__(self):
        pass

    def boost(self, training_data, testing_data, epochs):
        distribution = np.full((training_data.shape[0]), 1 / training_data.shape[0])
        training_prediction = np.repeat(0.0, training_data.shape[0])
        testing_prediction = np.repeat(0.0, testing_data.shape[0])
        round_error = []
        training_error_list = []
        for i in range(epochs):
            weak_classifier_used = weak_classifier_decision_stump_2()
            epsilon, column,  value = weak_classifier_used.fit(X = training_data[:,:-1], Y = training_data[:,-1], distributions = distribution)
            y_pred = weak_classifier_used.predict(value=value, column=column, X = training_data[:,:-1])
            round_error.append(epsilon)
            alpha = (0.5) * (np.log(1- epsilon) - np.log(epsilon))
            temp_distribution = (distribution * np.exp(-alpha * training_data[:,-1] * y_pred ))
            distribution = temp_distribution / np.sum(temp_distribution)
            training_prediction += (alpha * y_pred)
            y_pred_test = weak_classifier_used.predict(value = value, column = column, X = testing_data[:,:-1])
            testing_prediction += (alpha * y_pred_test)
            training_accuracy, training_error= self.calculate_error(training_prediction, training_data[:,-1])
            training_error_list.append(training_error)
            if(i % 10 == 0):
                print(i)
                print("training accuracy = ", training_accuracy)
        return testing_prediction, training_prediction

    def plot_error(self, round_error):
        list_indices = []
        for i in range(len(round_error)):
            list_indices.append(i)
        plt.plot(list_indices, round_error)
        plt.show()

    def plot_training_testing_error(self, error, i, flag):
        list_indices = []
        for i in range(i+1):
            list_indices.append(i)
        plt.plot(list_indices, error)
        if(flag):
            plt.ylabel('Testing error')
        else:
            plt.ylabel('Training error')
        plt.xlabel('rounds')
        plt.show()


    def calculate_error(self, prediction, label):
        temp_prediction = np.zeros(prediction.shape[0])
        temp_prediction[np.where(prediction <=0)] = -1
        temp_prediction[np.where(prediction > 0)] = 1
        return sklearn.metrics.accuracy_score(label, temp_prediction), sklearn.metrics.mean_squared_error(label, temp_prediction)

    def plot_ROC_curve(self, prediction, Y):
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(Y, prediction)
        roc_auc = sklearn.metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr)
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
        print("AUC = ", roc_auc)







