import pandas as pd
import numpy as np
from scipy.spatial import distance
X = np.array([[1,2,3],[4,5,6]])
print(distance.hamming(X[0], X[1]))
def read():
    data = pd.read_csv("./8newsgroup/train.trec/feature_matrix.txt", sep = "\s", header=None)
    c = []
read()