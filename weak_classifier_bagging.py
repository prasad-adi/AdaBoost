import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics


class weak_classifier_bagging:

    def classify_data(self, data):
        if (len(data) == 0):
            return -1
        label_column = data[:, -1]
        unique, count = np.unique(label_column, return_counts=True)
        return unique[np.argmax(count)]


    def separate_data(self, data, column, sortedIndices, split_Index):
        data_less = data[sortedIndices[:split_Index]]
        data_more = data[sortedIndices[split_Index:]]
        temp = data[:, column]
        value = temp[sortedIndices[split_Index - 1]]
        return value, data_less, data_more


    def calculate_entropy(self, data):
        label_column = data[:, -1]
        unique_elements, counts_elements = np.unique(label_column, return_counts=True)
        if (len(counts_elements) > 1):
            P1 = counts_elements[0] / len(data)
            P2 = counts_elements[1] / len(data)
            entropy = -(P1) * np.log2(P1) - (P2) * np.log2(P2)
        else:
            return 0
        return entropy


    def calculate_IG(self, data_less, data_more):
        total = len(data_less) + len(data_more)
        currentEntropy = self.calculate_entropy(np.concatenate((data_less, data_more), axis=0))
        IG = currentEntropy - ((len(data_less) / total) * self.calculate_entropy(data_less)
                               + (len(data_more) / total) * self.calculate_entropy(data_more))
        return IG


    def get_maximum_IG(self, data):
        minMse = 0
        best_column = -1
        best_value = -1
        best_data_less = np.array([])
        best_data_more = np.array([])
        for column in range(57):
            sortedIndices = np.argsort(data[:, column])
            split_Index = 1
            while (split_Index < (len(sortedIndices) - 5)):
                value, data_less, data_more = self.separate_data(data, column, sortedIndices, split_Index)
                mse = self.calculate_IG(data_less, data_more)
                if (mse > minMse):
                    minMse = mse
                    best_data_less = data_less
                    best_data_more = data_more
                    best_column = column
                    best_value = value
                split_Index = split_Index + 5
        return best_column, best_value, best_data_less, best_data_more

    def decisionTree(self,  data, min_samples, max_depth, current_depth=0):
        if (len(data) < min_samples or current_depth > max_depth):
            return self.classify_data(data)

        current_depth = current_depth + 1
        column, value, data_less, data_more = self.get_maximum_IG(data)
        classification = "{} <= {} left-{} right-{}".format(column, value, len(data_less), len(data_more))
        tree = {classification: []}

        left_side = self.decisionTree(data_less, min_samples, max_depth, current_depth)
        right_side = self.decisionTree(data_more, min_samples, max_depth, current_depth)

        # if len(left_side) < min_samples or len(right_side) <  min_samples:
        # return classify_data(data)

        tree[classification].append(left_side)
        tree[classification].append(right_side)
        return tree


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

    def predict(self, tree, data):
        y = []
        for i in range(data.shape[0]):
            y.append(self.classify_example(data[i, :], tree))
        return np.array(y)


