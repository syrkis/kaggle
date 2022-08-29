# main.py
#   kaggle main
# by: Noah Syrkis

# imports
from src import *
from comps import *
import numpy as np


# main
def main():
    args = get_args()
    config = get_config(args)

    model, (train_inputs, train_targets, test_inputs) = get_comp(args.comp)
    train_val_idxs = train_validation_split(len(train_targets))

    params = k_fold(model, train_inputs, train_targets, train_val_idxs, config, args)
    test_pred = make_submission_df(np.argmax(predict(params, test_inputs.to_numpy()), axis=1).astype(int), test_inputs.index)
    test_pred.to_csv("data/submission.csv", index=False)


if __name__ == '__main__':
    main()
