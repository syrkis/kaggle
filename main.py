# main.py
#   kaggle main
# by: Noah Syrkis

# imports
from src import *
from models import *
import numpy as np


# main
def main():
    args = get_args()

    scale = 0.1
    batch_size = 128
    layer_sizes = []
    num_epochs = 10
    step_size = 1e-3

    train_inputs, train_targets, train_n, test_inputs = get_data(args.comp)
    model = get_model(args.comp)
    batches = data_stream(train_inputs, train_targets, batch_size, k=10)
    params = train(model, batches, train_n//batch_size, scale, layer_sizes, num_epochs, step_size)
    """
    submission = predict(params, test_inputs)
    np.savetxt("data/submission.csv", submission, delimiter=",")
    """


if __name__ == '__main__':
    main()
