# data.py
#   kaggle dataset in jax
# by: Noah Syrkis

# imports
import numpy.random as npr
import pandas as pd
from src.utils import one_hot


# dataset function
def data_stream(inputs, targets, batch_size: int, k: int = None):
    targets = one_hot(targets, k) if k else targets
    num_batches = len(inputs) // batch_size
    rng = npr.RandomState(0)
    while True:
        perm = rng.permutation(len(inputs))
        for i in range(num_batches):
            batch_idx = perm[i * num_batches : (i + 1) * num_batches]
            yield inputs[batch_idx], targets[batch_idx]


# data reader and returner
def get_data(comp):
    if comp == 'tabular-playground-series-aug-2022':
        train_data = pd.get_dummies(pd.read_csv('data/train.csv', index_col='id'))
        train_inputs = train_data[train_data.columns[1:-1]].to_numpy()
        train_targets = train_data[train_data.columns[-1]].to_numpy()

        test_data = pd.get_dummies(pd.read_csv('data/test.csv', index_col='id'))
        test_inputs = test_data[test_data.columns[1:]].to_numpy()

        return train_inputs, train_targets, len(train_targets), test_inputs

    if comp == 'titanic':
        pass

