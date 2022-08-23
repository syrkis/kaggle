# models.py
#   get model for particular kaggle comp
# by: Noah Syrkis

# imports
from models import *

# return model function
def get_model(comp):
    if comp == 'tabular-playground-series-aug-2022':
        return tpsaug22
    if comp == 'titanic':
        return titanic

