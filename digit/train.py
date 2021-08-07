# trian.py
#   trains digit classifier
# by: Noah Syrkis

# imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler
from dataset import Dataset
from model import Model


# define training function
def train(model, dataset, optimizer, criterion, kf, epochs):
    
    # for every k-folds
    for train_idx, valid_idx in kf.split([i for i in range(len(dataset))]):

        # train sampler
        train_sampler = SubsetRandomSampler(train_idx) 

        # validation sapler
        valid_sampler = SubsetRandomSampler(valid_idx) 

        # train data loader
        trainer = DataLoader(dataset=dataset, batch_size=32, sampler=train_sampler)
        
        # valid data loader
        validator = DataLoader(dataset=dataset, batch_size=32, sampler=valid_sampler)

        # train model
        model.train()

        # total loss
        losses = 0

        # for every epoch
        for epoch in range(epochs):
    
            # for all data 
            for idx, (x, y) in enumerate(trainer):

                # resert gradient 
                optimizer.zero_grad()
        
                # make prediction
                p = model(x)

                # calculate loss
                loss = criterion(p, y)

                # make back prop
                loss.backward()

                # update weights  
                optimizer.step() 

                # add loss to total loss 
                losses += loss / x.shape[0]

            # print epoch loss
            print("train :", round((losses / len(trainer)).item(), 4), end = '\t\t')

            # validate performace
            validate(model, validator, criterion)


# validator function
def validate(model, validator, criterion):

    # evaluate model 
    model.eval()

    # total loss
    losses = 0

    # for all data
    for idx, (x, y) in enumerate(validator):

        # make prediction
        p = model(x)

        # calcualte loss
        loss = criterion(p, y)

        # add loss to total
        losses += loss / x.shape[0]
    
    # print epoch loss
    print("valid :", round((losses / len(validator)).item(), 4))



# dev call
def main():
    dataset = Dataset(train=True)
    model = Model()
    kf = KFold()
    optimizer = optim.Adam(model.parameters(), lr=0.001) 
    criterion = nn.CrossEntropyLoss()
    train(model, dataset, optimizer, criterion, kf, 5)
    torch.save(model.state_dict(), 'data/model')

if __name__ == '__main__':
    main()
