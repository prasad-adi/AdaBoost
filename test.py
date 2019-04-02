import pandas as pd
import numpy as np

x = np.array([[1,2,3],[4,5,6]])
indices = np.where(x[:,0] < 3)[0]
y = x[not indices]
c = []