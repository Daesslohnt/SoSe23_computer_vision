import os

import numpy as np

path = os.path.join("..", "daten", "X.npy")
X = np.load(path)
x_shape = X.shape

for i in range(len(X)):
    print(i)
    item = X[i].T
    x, y = item
    x_max, x_min = max(x), min(x)
    y_max, y_min = max(y), min(y)

    x = (x - x_min) / (x_max - x_min)
    y = (y - y_min) / (y_max - y_min)

    item[0] = x
    item[1] = y

    X[0] = item.T

assert x_shape == X.shape

np.save(path, X)
