import pandas as pd
import numpy as np



def read_and_process_data():
    with open("./8newsgroup/train.trec/feature_matrix.txt", "r") as f:
        contents = f.read()
        contents = contents.split("\n")
        X = np.zeros(1754).reshape(1, 1754)
        Y = []
        for i in range(len(contents) - 1):
            if(i % 1000 == 0):
                print(i)
            temp_array = np.zeros(1754)

            temp = contents[i].split(" ")
            Y.append(int(temp[0]))
            for j in range(1, len(temp) - 1):
                index_value = temp[j].split(":")
                temp_array[int(index_value[0])] = float(index_value[1])
            X = np.concatenate((X,temp_array.reshape(1, 1754)), axis = 0)
    np.savetxt("./X.txt", X)
    np.savetxt("./Y.txt", X = np.array(Y))
    return X, np.array(Y)

def generate_class_codes(k):
    np.



#X, Y = read_and_process_data()
X_train = np.loadtxt("./X.txt")
Y_train = np.loadtxt("./Y.txt")
X_train = X_train[1:]
c =[]
class_codes = generate_class_codes(8)