# titanic.py
#   baseline sollution to titanic
# by: Noah Syrkis

# imports
import pandas as pd
from sklearn.svm import SVC

# setup
df = pd.read_csv('train.csv').dropna()
y = df[df.columns[1]]
X = df[df.columns[[2,5,6,7]]]

# training
clf = SVC()
clf.fit(X, y)

# predict
df = pd.read_csv('test.csv').dropna()
X = df[df.columns[[1,4,5,6]]]
p = clf.predict(X)
print(p)
