# data.py
#   kaggle dataset in jax
# by: Noah Syrkis

# imports
import numpy.random as npr


# dataset function
def data_stream(inputs, targets, batch_size: int):
    num_batches = len(inputs) // batch_size
    rng = npr.RandomState(0)
    while True:
        perm = rng.permutation(len(inputs))
        for i in range(num_batches):
            batch_idx = perm[i * num_batches : (i + 1) * num_batches]
            yield inputs[batch_idx], targets[batch_idx]

