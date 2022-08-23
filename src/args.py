# args.py
#   read command line arguments
# by: Noah Syrkis

# imports
import argparse


# get arguments
def get_args():
    parser = argparse.ArgumentParser(description='Run Kaggle Competition')
    parser.add_argument('--comp', type=str, help='Kaggle competition id')
    args = parser.parse_args()
    return args
