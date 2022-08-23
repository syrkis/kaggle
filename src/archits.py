# archits.py
#   model architectures for kaggle
# by: Noah Syrkis

# imports
import jax.numpy as jpn


# fully connected
def fully_connected(params, inputs):
    activations = inputs
    for w, b in params[:-1]:
        outputs = jnp.dot(activations, w) + b
        activations = jnp.tanh(activations)

    # output layer
    final_w, final_b = params[-1]
    logits = jnp.dot(activations, final_w) + final_b
    return logits # softmax????

