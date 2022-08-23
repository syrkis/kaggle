# comps.py
#   competition specific code
# by: Noah Syrkis

# imports
from comps import *
import pandas as pd


# data preproccsing
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


# return model function
def get_model(comp):
    if comp == 'tabular-playground-series-aug-2022':
        return lambda params, inputs: tpsaug22(params, inputs)
    if comp == 'titanic':
        return lambda params, inputs: titanic(params, inputs)

