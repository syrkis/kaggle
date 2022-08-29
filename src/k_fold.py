# kfold.py
#   kaggle k-fold validation
# by: Noah Syrkis

# imports
import wandb
from src.data import data_stream
from src.train import train


# train
def k_fold(model, train_inputs, train_targets, train_validation_idxs, args):
    for train_idxs, val_idxs in train_validation_idxs:
        train_batches = data_stream(train_inputs[train_idxs], train_targets[train_idxs], batch_size, k=10)
        val_batches = data_stream(train_inputs[val_idxs], train_targets[val_idxs], batch_size, k=10)
        params = model.train(model, train_batches, val_batches)

