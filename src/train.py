# train.py
#   train jax model for kaggle
# by: Noah Syrkis

# imports
from src.utils import loss

import numpy.random as npr
from jax import grad, jit

import time


# train function
def train(model, batches, num_batches, scale, layer_sizes, num_epochs, step_size):
    params = init_params(scale, layer_sizes)

    for epoch in range(num_epochs):
        start_time = time.time()
        for _ in range(num_batches):
            params = update(model, params, next(batches), step_size)
        epoch_time = time.time() - start_time

    return params


# update parameters with back propagation
def update(model, params, batch, step_size):
    grads = grad(loss)(params, batch)
    return [(w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(params, grads)]


# initialize random, normally distributed weights
def init_params(scale, layer_sizes, rng=npr.RandomState(0)):
    params = [(scale * rng.randn(m, n), scale * rng.randn(n))
              for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

    return params
