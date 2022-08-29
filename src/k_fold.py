# k_fold.py
#   kaggle k-fold validation
# by: Noah Syrkis

# imports
import wandb
from src.data import data_stream
from src.train import train


# train
def k_fold(model, train_inputs, train_targets, train_val_idxs, config, args):
    for train_idxs, val_idxs in train_val_idxs:
        num_train_batches = len(train_idxs) // config['batch_size']
        train_batches = data_stream(train_inputs.loc[train_idxs], train_targets.loc[train_idxs], config['batch_size'], k=2)
        val_batches = data_stream(train_inputs.loc[val_idxs], train_targets.loc[val_idxs], config['batch_size'], k=2)
        params = train(model, train_batches, num_train_batches, val_batches, config, args)
    return params
