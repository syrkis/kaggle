# utils.py
#   helper functions
# by: Noah Syrkis

# imports
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpy as np
import json


# prediction function for fully connected linear layers
def predict(params, inputs):
    activations = inputs
    for w, b in params[:-1]:
        outputs = jnp.dot(activations, w) + b
        activations = jnp.tanh(outputs)

    # output layer
    final_w, final_b = params[-1]
    logits = jnp.dot(activations, final_w) + final_b
    return logits - logsumexp(logits, axis=1, keepdims=True) # why this


# cross entropy loss
def loss(params, model, batch):
    inputs, targets = batch
    preds = model(params, inputs)
    return - jnp.mean(jnp.sum(preds * targets, axis=1))


# accuracy
def accuracy(params, batch):
    inputs, targets = batch
    pred = jnp.argmax(predict(params, inputs), axis=1)
    return jnp.mean(pred == targets)


# one hot encode vector
def one_hot(targets, k, dtype=np.float32):
    return np.array(targets[:, None] == np.arange(k), dtype)


# standardise dataframe
def z_score(train_df, test_df):
    mu = np.mean(train_df, axis=0)
    sigma = np.std(train_df, axis=0)
    standard_train = (train_df - mu) / sigma
    standard_test = (test_df - mu) / sigma
    return standard_train, standard_test


# get params file with model parameters
def get_config(args):
    with open('config.json', 'r') as f:
        config = json.load(f)
    return config[args.comp]















