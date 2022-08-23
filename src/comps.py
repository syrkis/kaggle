# comps.py
#   competition specific code
# by: Noah Syrkis

# imports
import pandas as pd

def get_data(comp):
    if comp == 'tabular-playground-series-aug-2022':
        train_data = pd.get_dummies(pd.read_csv('data/train.csv', index_col='id'))
        train_inputs = train_data[train_data.columns[1:-1]].to_numpy()
        train_targets = train_data[train_data.columns[-1]].to_numpy()

        test_data = pd.get_dummies(pd.read_csv('data/test.csv', index_col='id'))
        test_inputs = test_data[test_data.columns[1:]].to_numpy()

        return train_inputs, train_targets, test_inputs

    if comp == 'titanic':
        pass

