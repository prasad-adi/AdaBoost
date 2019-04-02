import numpy as np
from weak_classifier_regression import Weak_classifier_Regression



class Gradient_Boosting:
    def boost(self, X_train, Y_train, X_test, Y_test, epochs):
        training_prediction = np.repeat(np.mean(Y_train), X_train.shape[0])
        testing_prediction = np.repeat(np.mean(Y_train), X_test.shape[0])
        for i in range(epochs):
            y_train_temp = Y_train - training_prediction
            classifier = Weak_classifier_Regression()
            y_pred_train, y_pred_test = classifier.fit(X_train, y_train_temp, X_test, 1)
            training_prediction += y_pred_train
            testing_prediction +=y_pred_test
            self._calculate_mse(training_prediction, Y_train, testing_prediction, Y_test)


    def _calculate_mse(self, y_pred_train, Y_train, y_pred_test, Y_test):
        training_mse = np.sum(np.square(Y_train - y_pred_train)) / len(Y_train)
        testing_mse = np.sum(np.square(Y_test - y_pred_test)) / len(Y_test)
        print("training mse = ", training_mse)
        print("testing_mse = ", testing_mse)







