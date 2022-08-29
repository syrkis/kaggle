# data.py
#   kaggle dataset in jax
# by: Noah Syrkis

# imports
import numpy.random as npr
import pandas as pd
import numpy as np
from src.utils import one_hot


# dataset function
def data_stream(inputs, targets, batch_size: int, k: int = None): # TODO: val
    inputs = inputs.to_numpy()
    targets = targets.to_numpy()
    targets = one_hot(targets, k) if k else targets
    num_batches = len(inputs) // batch_size
    rng = npr.RandomState(0)
    while True:
        perm = rng.permutation(len(inputs))
        for i in range(num_batches):
            batch_idx = perm[i * num_batches: (i + 1) * num_batches]
            yield inputs[batch_idx], targets[batch_idx]


# train validation fold splits
def train_validation_split(num_samples, k=2):
    rng = npr.RandomState(0)
    perm = rng.permutation(num_samples)
    idxs = []
    for i in range(k):
        val_idxs = perm[i * (num_samples // k): (i + 1) * (num_samples // k)]
        train_idxs = [idx for idx in perm if idx not in val_idxs]
        idxs.append((train_idxs, val_idxs))
    return idxs


# numpy to pandas with id for submission
def make_submission_df(pred, idxs):
    df = pd.DataFrame()
    df['id'] = idxs
    df['failure'] = pred
    return df
