import numpy as np
from weak_classifier_regression import Weak_classifier_Regression
from sklearn.tree import DecisionTreeRegressor
import sklearn


class Gradient_Boosting_BuiltIn:
    def boost(self, X_train, Y_train, X_test, Y_test, epochs):
        training_prediction = np.repeat(np.mean(Y_train), X_train.shape[0])
        testing_prediction = np.repeat(np.mean(Y_train), X_test.shape[0])
        for i in range(epochs):
            y_train_temp = Y_train - training_prediction
            weak_classifier = DecisionTreeRegressor(max_depth=2)
            weak_classifier.fit(X_train, y_train_temp)
            y_pred_train = weak_classifier.predict(X_train)
            y_pred_test = weak_classifier.predict(X_test)
            training_prediction += y_pred_train
            testing_prediction += y_pred_test
            print("training_mse = ",sklearn.metrics.mean_squared_error(Y_train, training_prediction))
            print("testing_mse = ", sklearn.metrics.mean_squared_error(Y_test, testing_prediction))