import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset_train = pd.read_csv('housing_train.txt', sep='\s+', header=None)
X_train = dataset_train.iloc[:, :-1].values
y_train = dataset_train.iloc[:, 13].values
dataset = dataset_train.values

dataset_test = pd.read_csv('housing_test.txt', sep='\s+', header=None)
X_test = dataset_test.iloc[:, :-1].values
y_test = dataset_test.iloc[:, 13].values
test = dataset_test.values

for i in range(13):
    temp = dataset[:, i]
    temp1 = test[:, i]
    mean = np.mean(temp)
    std = np.std(temp)
    dataset[:, i] = (temp - mean) / std
    test[:, i] = (temp1 - mean) / std


# Classify data
def classify_data(data):
    if (len(data) == 0):
        return 0
    return np.mean(data[:, -1])


def separate_data(data, column, sortedIndices, split_Index):
    data_less = data[sortedIndices[:split_Index]]
    data_more = data[sortedIndices[split_Index:]]
    temp = data[:, column]
    value = temp[sortedIndices[split_Index]]
    return value, data_less, data_more


def calculate_mse(data_less, data_more):
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


def get_minimum_mse(data):
    minMse = 999
    # currentMse = calculate_mse(data)
    best_column = -1
    best_value = -1
    best_data_less = np.array([])
    best_data_more = np.array([])
    for column in range(13):
        sortedIndices = np.argsort(data[:, column])
        split_Index = 1
        while (split_Index < (len(sortedIndices) - 5)):
            value, data_less, data_more = separate_data(data, column, sortedIndices, split_Index)
            mse = calculate_mse(data_less, data_more)
            if (mse < minMse):
                minMse = mse
                best_data_less = data_less
                best_data_more = data_more
                best_column = column
                best_value = value
            split_Index = split_Index + 1
    return best_column, best_value, best_data_less, best_data_more


def decisionTree(data, min_samples, max_depth, current_depth=0):
    if (len(data) < min_samples or current_depth > max_depth):
        return classify_data(data)

    current_depth = current_depth + 1
    column, value, data_less, data_more = get_minimum_mse(data)
    classification = "{} <= {} left-{} right-{}".format(column, value, len(data_less), len(data_more))
    tree = {classification: []}
    left_side = decisionTree(data_less, min_samples, max_depth, current_depth)
    right_side = decisionTree(data_more, min_samples, max_depth, current_depth)
    if left_side == right_side:
        tree = left_side
    else:
        tree[classification].append(left_side)
        tree[classification].append(right_side)
    return tree


# 1,4
from pprint import pprint

tree = decisionTree(dataset, 1, 2, 0)


def classify_example(example, tree):
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
        return classify_example(example, residual_tree)


y = []
for i in range(74):
    y.append(classify_example(test[i, :], tree))
Y = np.array(y)
print(np.sum(np.square(Y - y_test)) / len(y_test))