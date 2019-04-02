import numpy as np

np.random.seed(10)


class weak_classifier_decision_stump_2:


    def fit(self, X, Y):
        thresholds = self.__getThresholds(X)
        best_column = 0
        best_value = 0
        max_mse = 0
        best_error = 0
        for i in range(len(thresholds)):
            for j in range(len(thresholds[i])):
                y_pred = self.predict(column = i, value = thresholds[i][j], X = X)
                error = self.__calculate_mse(y_pred, Y)
                mse = abs(0.5 - error)
                if(mse >= max_mse):
                    max_mse = mse
                    best_error = error
                    best_column = i
                    best_value = thresholds[i][j]
        return best_error, best_column, best_value

    def predict(self, column, value, X):
        y_pred = np.where(X[:,column] <= value, -1,1)
        return y_pred

    def __calculate_mse(self, y_pred, Y):
         return np.sum(np.square(y_pred - Y)) / Y.shape[0]


    def  __getThresholds(self, X):
        thresholds = []
        for i in range(X.shape[1]):
            column_thresholds = []
            temp_thresholds = np.unique(X[:,i])
            column_thresholds.append(temp_thresholds[0] - 1)
            for i in range(1, len(temp_thresholds)):
                column_thresholds.append((temp_thresholds[i] + temp_thresholds[i-1]) / 2)
            column_thresholds.append(temp_thresholds[-1] + 1)
            thresholds.append(column_thresholds)
        return thresholds


