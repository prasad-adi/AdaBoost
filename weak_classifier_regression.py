import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Weak_classifier_Regression:
    def fit(self, X_train, Y_train, X_test, max_depth):
        tree = self.__decision_tree(np.c_[X_train, Y_train], max_depth, 0)
        y_pred_train = self.predict(X_train, tree)
        y_pred_test = self.predict(X_test, tree)
        return y_pred_train, y_pred_test


    def __decision_tree(self, data, max_depth, current_depth):
        if (current_depth >= max_depth):
            return self.classify_data(data)
        current_depth = current_depth + 1
        column, value, data_less, data_more = self.get_minimum_mse(data)
        classification = "{} <= {} left-{} right-{}".format(column, value, len(data_less), len(data_more))
        tree = {classification: []}
        if(data.size != 0):
            left_side = self.__decision_tree(data_less, max_depth, current_depth)
            right_side = self.__decision_tree(data_more, max_depth, current_depth)
            if left_side == right_side:
                tree = left_side
            else:
                tree[classification].append(left_side)
                tree[classification].append(right_side)
        return tree

    def classify_data(self, data):
        if (len(data) == 0):
            return 0
        return np.mean(data[:, -1])

    def calculate_mse(self, data_less, data_more):
        mse = 0
        data = data_less[:, -1]
        mean = np.mean(data)
        X = np.square(mean - data)
        mse = mse + np.sum(X)

        data = data_more[:, -1]
        mean = np.mean(data)
        X = np.square(mean - data)
        mse = mse + np.sum(X)
        return mse / (len(data_less) + len(data_more))

    def get_minimum_mse(self, data):
        minMse = 999
        # currentMse = calculate_mse(data)
        best_column = -1
        best_value = -1
        best_data_less = np.array([])
        best_data_more = np.array([])
        for column in range(data.shape[1] - 1):
            minMse, best_column, best_value, best_data_less, best_data_more = self.get_best_values(data, column, minMse, best_column, best_value,best_data_less,best_data_more)
        return best_column, best_value, best_data_less, best_data_more

    def get_best_values(self, data, column, minMse,  best_column, best_value, best_data_less, best_data_more):
        sortedIndices = np.argsort(data[:, column])
        split_Index = 1
        while (split_Index < (len(sortedIndices) - 5)):
            value, data_less, data_more = self.separate_data(data, column, sortedIndices, split_Index)
            mse = self.calculate_mse(data_less, data_more)
            if (mse < minMse):
                minMse = mse
                best_data_less = data_less
                best_data_more = data_more
                best_column = column
                best_value = value
            split_Index = split_Index + 1
        return minMse, best_column, best_value, best_data_less, best_data_more


    def separate_data(self, data, column, sortedIndices, split_Index):
        data_less = data[sortedIndices[:split_Index]]
        data_more = data[sortedIndices[split_Index:]]
        temp = data[:, column]
        value = temp[sortedIndices[split_Index]]
        return value, data_less, data_more

    def classify_example(self, example, tree):
        question = list(tree.keys())[0]
        feature_name, comparison_operator, value, left, right = question.split(" ")

        # ask question
        if example[int(feature_name)] <= float(value):
            answer = tree[question][0]
        else:
            answer = tree[question][1]

        # base case
        if not isinstance(answer, dict):
            return answer

        # recursive part
        else:
            residual_tree = answer
            return self.classify_example(example, residual_tree)

    def predict(self, X, tree):
        y = []
        for i in range(X.shape[0]):
            y.append(self.classify_example(X[i, :], tree))

        return np.array(y)



