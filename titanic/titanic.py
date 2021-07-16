# titanic.py
#   vanilla pytorch nn titanic
# by: Noah Syrkis

# imports
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import pandas as pd
import numpy as np
from icecream import ic

# net class
class Net(nn.Module):
    """"""

    def __init__(self, input_dim):
        self.fc1 = nn.Linear(input_dim, 2)
    
    def forward(self, x):
        x = self.fc1(x)
        return x

# data read
df = pd.read_csv('data/train.csv')
y = torch.tensor(df['Survived'].values)
X = df[df.columns[[2, 6, 7, 9]]]
X = torch.tensor(np.array(X, dtype=np.float32))

# train loop
M = Net(4)
L = nn.BCELoss()
J = optim.Adam(params=M.parameters(), lr=0.001)

for _ in range(100):
    J.zero_grad() 
