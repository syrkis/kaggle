# comps.py
#   get model and data for particular kaggle comp
# by: Noah Syrkis

# imports
from comps import *

# return model function
def get_comp(comp):
    if comp == 'tabular-playground-series-aug-2022':
        return tpsaug22.model, tpsaug22.get_data()
    if comp == 'titanic':
        return titanic.model, titanic.get_data()

