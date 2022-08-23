# utils.py
#   helper functions
# by: Noah Syrkis

# imports
import jax.numpy as jnp
import numpy as np


# prediction function for fully connected linear layers
def predict(params, inputs):
    activations = inputs
    for w, b in params[:-1]:
        outputs = jnp.dot(activations, w) + b
        activations = jnp.tanh(outputs)

    # output layer
    final_w, final_b = params[-1]
    logits = jnp.dot(activations, final_w) + final_b
    return logits


# cross entropy loss
def loss(model, params, batch):
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

