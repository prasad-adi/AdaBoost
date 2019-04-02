import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics

dataset_train = pd.read_csv('./spambase.data.txt', header=None)
X_train = dataset_train.iloc[:, :-1].values
y_train = dataset_train.iloc[:, 57].values
dataset = dataset_train.values

np.random.shuffle(dataset)

for i in range(57):
    temp = dataset[:, i]
    dataset[:, i] = (temp - np.mean(temp)) / np.std(temp)


def classify_data(data):
    if (len(data) == 0):
        return 0
    label_column = data[:, -1]
    unique, count = np.unique(label_column, return_counts=True)
    return unique[np.argmax(count)]


def separate_data(data, column, sortedIndices, split_Index):
    data_less = data[sortedIndices[:split_Index]]
    data_more = data[sortedIndices[split_Index:]]
    temp = data[:, column]
    value = temp[sortedIndices[split_Index - 1]]
    return value, data_less, data_more


def calculate_entropy(data):
    label_column = data[:, -1]
    unique_elements, counts_elements = np.unique(label_column, return_counts=True)
    if (len(counts_elements) > 1):
        P1 = counts_elements[0] / len(data)
        P2 = counts_elements[1] / len(data)
        entropy = -(P1) * np.log2(P1) - (P2) * np.log2(P2)
    else:
        return 0
    return entropy


def calculate_IG(data_less, data_more):
    total = len(data_less) + len(data_more)
    currentEntropy = calculate_entropy(np.concatenate((data_less, data_more), axis=0))
    IG = currentEntropy - ((len(data_less) / total) * calculate_entropy(data_less)
                           + (len(data_more) / total) * calculate_entropy(data_more))
    return IG


def get_maximum_IG(data):
    minMse = 0
    best_column = -1
    best_value = -1
    best_data_less = np.array([])
    best_data_more = np.array([])
    for column in range(57):
        sortedIndices = np.argsort(data[:, column])
        split_Index = 1
        while (split_Index < (len(sortedIndices) - 5)):
            value, data_less, data_more = separate_data(data, column, sortedIndices, split_Index)
            mse = calculate_IG(data_less, data_more)
            if (mse > minMse):
                minMse = mse
                best_data_less = data_less
                best_data_more = data_more
                best_column = column
                best_value = value
            split_Index = split_Index + 5
    return best_column, best_value, best_data_less, best_data_more


accuracy = 0


def decisionTree(data, min_samples, max_depth, current_depth=0):
    if (len(data) < min_samples or current_depth > max_depth):
        return classify_data(data)

    current_depth = current_depth + 1
    column, value, data_less, data_more = get_maximum_IG(data)
    classification = "{} <= {} left-{} right-{}".format(column, value, len(data_less), len(data_more))
    tree = {classification: []}

    left_side = decisionTree(data_less, min_samples, max_depth, current_depth)
    right_side = decisionTree(data_more, min_samples, max_depth, current_depth)

    # if len(left_side) < min_samples or len(right_side) <  min_samples:
    # return classify_data(data)

    tree[classification].append(left_side)
    tree[classification].append(right_side)
    return tree


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


from pprint import pprint

k_split = 2
k = [i for i in range(k_split)]
n = np.array_split(dataset, k_split)
error = 0
accuracy1 = 0
X = 0
for i in range(k_split):
    X_test = n[i]
    y_test = X_test[:, -1]
    counter = 0
    for j in range(k_split):
        if (j != i):
            if (counter == 0):
                X = n[j]
                counter += 1
            elif(counter > 0):
                X = np.concatenate((X, n[j]), axis=0)
    tree = decisionTree(X, 8, 0, 0)
    pprint(tree)

    y = []
    for i in range(len(X_test)):
        y.append(classify_example(X_test[i, :], tree))
    Y = np.array(y)

    error = error + np.sum(np.square(Y - y_test)) / len(y_test)
    print("error = ", np.sum(np.square(Y - y_test)) / len(y_test))
    accuracy = accuracy + sklearn.metrics.accuracy_score(y_test, Y)

    print("accuracy = ", sklearn.metrics.accuracy_score(y_test, Y))

print(error / k_split)
print(accuracy / k_split)