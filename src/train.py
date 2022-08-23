# train.py
#   train jax model for kaggle
# by: Noah Syrkis

# imports
import numpy.random as npr
from jax import grad, jit

import time


# train function
def train(batches, num_batches, scale, layer_sizes, num_epochs, step_size):
    params = init_params(scale, layer_sizes)

    for epoch in num_epochs:
        start_time = time.time()
        for _ in range(num_batches):
            params = update(params, next(batches), step_size)
        epoch_time = time.time() - start_time


# update parameters with back propagation



# initialize random, normally distributed weights
def init_params(scale, layer_sizes, rng=RandomState(0)):
    params = [(scale * rng.randn(m, n), scale * rng.randn(n))
              for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]
