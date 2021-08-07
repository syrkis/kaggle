# titanic.py
#   vanilla pytorch nn titanic
# by: Noah Syrkis

# imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from icecream import ic
from matplotlib import pyplot as plt
from sklearn import tree


# loading data
train = pd.read_csv("data/train.csv")
train.drop(['Cabin'], 1, inplace=True)
train = train.dropna()
y = train['Survived']
train.drop(['Survived', 'PassengerId', 'Name', 'Ticket'], 1, inplace=True)
x = pd.get_dummies(train)

dtc = tree.DecisionTreeClassifier()
dtc.fit(x, y)

test = pd.read_csv('data/test.csv')
ids = test[['PassengerId']]
test.drop(['PassengerId', 'Cabin', 'Name', 'Ticket'], 1, inplace=True)
test.fillna(2, inplace=True)
test = pd.get_dummies(test)

pred = dtc.predict(test)
results = ids.assign(Survived = pred)
results.to_csv('results.csv', index=False)


