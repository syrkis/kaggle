# main.py
#   kaggle main
# by: Noah Syrkis

# imports
from src import *
import numpy as np


# main
def main():
    train_inputs, train_targets, test_inputs = get_data(args.comp)
    batches = data_stream(train_inputs, train_targets)
    params = train(batches)
    submission = predict(params, test_inputs)
    np.savetxt("data/submission.csv", submission, delimiter=",")


if __name__ == '__main__':
    main()
