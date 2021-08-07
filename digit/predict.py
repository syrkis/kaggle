# predict.py
#   makes digit predictions
# by: Noah Syrkis

# imports
import torch
from torch.utils.data import DataLoader
from model import Model
from dataset import Dataset
from icecream import ic


# predict function
def predict(model, loader):
    with open('data/submission.csv', 'w') as f:
        f.write('ImageId,Label\n') 
        i = 1
        for x in loader:
            p = model(x) 
            p = torch.argmax(p, dim=1)
            for idx in range(len(p)): 
                f.write(f"{i},{p[idx].item()}\n")
                i += 1


# script call
def main():
    model = Model()
    model.load_state_dict(torch.load('data/model'))
    model.eval()
    dataset = Dataset(train=False)
    loader = DataLoader(dataset=dataset, batch_size=2 ** 9)
    predict(model, loader)

if __name__ == '__main__':
    main()
