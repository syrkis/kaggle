# data.py
#   kaggle dataset in jax
# by: Noah Syrkis

# imports
import numpy.random as npr


# dataset function
def data_stream(data, batch_size: int):
    inputs = data[data.columns[1:-1]].to_numpy()
    targets = data[data.columns[-1]].to_numpy()
    num_batches = len(data) // batch_size
    rng = npr.RandomState(0)
    while True:
        perm = rng.permutation(len(data))
        for i in range(num_batches):
            batch_idx = perm[i * num_batches : (i + 1) * num_batches]
            yield inputs[batch_idx], targets[batch_idx]


