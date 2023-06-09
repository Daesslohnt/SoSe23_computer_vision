import numpy as np
import os
import re

DIR = os.path.join("..", "daten")

X_files = [os.path.join(DIR, name) for name in os.listdir(DIR) if re.match('X[a-zA-Z-\.npy]', name)]
y_files = [os.path.join(DIR, name) for name in os.listdir(DIR) if re.match('y[a-zA-Z-\.npy]', name)]

X_data = [np.load(path) for path in X_files]
X_data = np.concatenate(X_data)
np.save(os.path.join(DIR, "X.npy"), X_data)

y_data = [np.load(path) for path in y_files]
y_data = np.concatenate(y_data)
np.save(os.path.join(DIR, "y.npy"), y_data)