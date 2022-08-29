# train.py
#   train jax model for kaggle
# by: Noah Syrkis

# imports
from src.utils import loss

import numpy.random as npr
from jax import grad, jit

import wandb


# train function
def train(model, train_batches, num_train_batches, val_batches, config, args):

    wandb.init(entity='syrkis', project="kaggle", name=args.comp, reinit=False)
    params = init_params(config['scale'], config['layer_sizes'])
    for epoch in range(config['num_epochs']):
        for _ in range(num_train_batches):

            val_batch = next(val_batches)
            train_batch = next(train_batches)

            val_loss = loss(params, model, val_batch)
            train_loss = loss(params, model, train_batch)

            params = update(params, model, train_batch, config['step_size'])
            wandb.log({"Train loss": train_loss, "Val loss": val_loss})

    return params


# update parameters with back propagation
def update(params, model, batch, step_size):
    grads = grad(loss)(params, model, batch)
    return [(w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(params, grads)]


# initialize random, normally distributed weights
def init_params(scale, layer_sizes, rng=npr.RandomState(0)):
    params = [(scale * rng.randn(m, n), scale * rng.randn(n))
              for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

    return params
