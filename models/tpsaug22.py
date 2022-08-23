# tpsaug22.py
#   kaggle comp model
# by: Noah Syrkis

# imports
import jax.numpy as jnp


def tpsaug22(params, inputs):
    activations = inputs
    for w, b in params[:-1]:
        outputs = jnp.dot(activations, w) + b
        activations = jnp.tanh(outputs)

    # output layer
    final_w, final_b = params[-1]
    logits = jnp.dot(activations, final_w) + final_b
    return logits
