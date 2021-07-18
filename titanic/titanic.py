# titanic.py
#   vanilla pytorch nn titanic
# by: Noah Syrkis

# imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from icecream import ic
from matplotlib import pyplot as plt


# loading data
class TitanicDataset(Dataset): 
    
    def __init__(self, train = True):
        self.train = train
        self.y = None
        features = 'Pclass,Sex,Age,SibSp,Parch,Fare'.split(',')
        if not train:
            self.xy = pd.read_csv('./data/test.csv', quotechar='"')
        else: 
            self.xy = pd.read_csv('./data/train.csv', quotechar='"')
        self.x = self.xy[features]
        Sex = np.array(self.x.Sex == 'male') * 1
        self.x['Sex'] = Sex
        if train:
            target = 'Survived'
            self.y = self.xy[[target]]
            self.y = torch.from_numpy(np.array(self.y).astype(np.float32))
        self.n_samples = self.xy.shape[0]
        self.x = torch.nan_to_num(torch.from_numpy(np.array(self.x).astype(np.float32)))
    
    def __getitem__(self, idx):
            if self.train:
                return self.x[idx], self.y[idx]
            else:
                return self.x[idx]

    def __len__(self):
        return self.n_samples

# neural net
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6, 3)
        self.fc2 = nn.Linear(3, 1)    
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


dataset = TitanicDataset()
dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=4)
dataiter = iter(dataloader)

L = nn.BCEWithLogitsLoss()
M = Net()
J = optim.Adam(params=M.parameters(), lr=0.001)
A = nn.Sigmoid()
n_epochs = 2

for epoch in range(n_epochs):
    for i, (x, t) in enumerate(dataloader):
        J.zero_grad()
        pred = M(x)
        loss = L(pred, t)
        loss.backward()
        J.step()


dataset = TitanicDataset(train = False)
pred = M(dataset[:])
ic(pred.shape)
ic(torch.sum(A(pred) > .5))
