# tpsaug22.py
#   kaggle comp model
# by: Noah Syrkis

# imports
from src.utils import z_score, one_hot
from src.train import init_params

import jax.numpy as jnp
import pandas as pd
import numpy as np


def model(params, inputs):
    """
    :param params:
    :param inputs:
    :return: outputs """
    activations = inputs
    for w, b in params[:-1]:
        outputs = jnp.dot(activations, w) + b
        activations = jnp.tanh(outputs)

    # output layer
    final_w, final_b = params[-1]
    logits = jnp.dot(activations, final_w) + final_b
    return logits


def get_data():
    # load test and train data into pandas
    train_data = pd.read_csv('data/train.csv', index_col='id')
    test_inputs = pd.read_csv('data/test.csv', index_col='id')

    train_target = train_data[train_data.columns[-1]]
    train_inputs = train_data[train_data.columns[:-1]]

    # drop categorical columns with little overlap
    train_inputs = train_inputs.drop(train_inputs.columns[[0,3]], axis=1)
    test_inputs = test_inputs.drop(test_inputs.columns[[0,3]], axis=1)

    # get dummie variable for reamining categorical column
    train_inputs = pd.get_dummies(train_inputs)
    test_inputs = pd.get_dummies(test_inputs)

    # standardise
    train_inputs, test_inputs = z_score(train_inputs, test_inputs)
    train_inputs, test_inputs = train_inputs.fillna(0), test_inputs.fillna(0)

    return train_inputs, train_target, test_inputs


