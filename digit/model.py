# model.py
#   pytorch model for digit classificaiton
# by: Noah Syrkis

# imports
import torch
import torch.nn as nn
import torch.nn.functional as F


# model class
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # declar convolutional layers
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        # dropout layer
        self.do = nn.Dropout(p=0.5)

        # declare linear layers
        self.fc1 = nn.Linear(2 ** 8, 140)
        self.fc2 = nn.Linear(140, 70)
        self.fc3 = nn.Linear(70, 10) 

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = self.do(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x 


# dev calls
def main():
    model = Model()
    
if __name__ == '__main__':
    main()
