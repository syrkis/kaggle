# imports
import pandas as pd
import numpy as np
import random
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# reads
train = pd.read_csv('train.csv')
mask = train.columns[train.dtypes == 'int64']
X, y = train[mask[:-1]], train[mask[-1]]
X_train, X_test, y_train, y_test = train_test_split(X, y)
r = np.random.randint(min(y_train), max(y_train), y_test.shape[0])

# regres
clf = SVR()
clf.fit(X_train, y_train)
p = clf.predict(X_test)
y = np.array(y_test)
print(mean_squared_error(y, r) / mean_squared_error(y, p))

test = pd.read_csv('test.csv')
X = test[mask[:-1]].dropna()
P = clf.predict(X)
