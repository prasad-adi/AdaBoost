import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt
from weak_classifier_random import weak_classifier_random
np.random.seed(42)

class AdaBoostRandom:
    def __init__(self):
        pass

    def boost(self, training_data, testing_data, epochs):
        distribution = np.full((training_data.shape[0]), 1 / training_data.shape[0])
        training_prediction = np.repeat(0.0, training_data.shape[0])
        testing_prediction = np.repeat(0.0, testing_data.shape[0])
        round_error = []
        training_error_list = []
        testing_error_list = []
        for i in range(epochs):
            weak_classifier = weak_classifier_random(training_data, distribution)
            column,value = weak_classifier.build_stump()
            y_pred = weak_classifier.predict(training_data)
            wrong_indices = np.where(y_pred != training_data[:,-1])[0]
            epsilon = np.sum(distribution[wrong_indices])
            round_error.append(epsilon)
            alpha = (0.5) * (np.log(1- epsilon) - np.log(epsilon))
            temp_distribution = (distribution * np.exp(-alpha * training_data[:,-1] * y_pred ))
            distribution = temp_distribution / np.sum(temp_distribution)
            training_prediction += (alpha * y_pred)
            y_pred_test = weak_classifier.predict(testing_data)
            testing_prediction += (alpha * y_pred_test)
            training_accuracy, training_error= self.calculate_error(training_prediction, training_data[:,-1])
            training_error_list.append(training_error)
            testing_accuracy, testing_error = self.calculate_error(testing_prediction, testing_data[:,-1])
            testing_error_list.append(testing_error)

            if(i % 100 == 0):
                print("training accuracy = ", training_accuracy)
                print("testing_accuracy = ", testing_accuracy)
                #if(i > 0):
                    #self.plot_training_testing_error(training_error, i, 0)
                    #self.plot_training_testing_error(testing_error, i, 1)
        return testing_prediction, training_prediction

    def plot_training_testing_error(self, error, i, flag):
        list_indices = []
        for i in range(i):
            list_indices.append(i)
        plt.plot(list_indices, error)
        if (flag):
            plt.ylabel('Testing error')
        else:
            plt.ylabel('Training error')
        plt.xlabel('rounds')
        plt.show()

    def plot_error(self, round_error):
        list_indices = []
        for i in range(len(round_error)):
            list_indices.append(i)
        plt.plot(list_indices, round_error)
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







