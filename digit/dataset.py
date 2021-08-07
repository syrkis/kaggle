# dataset.py
#   reads digit data
# by: Noah Syrkis

# imports
import torch
import numpy as np


# define dataset class
class Dataset(torch.utils.data.Dataset):

    # init dataset
    def __init__(self, train=True):

        # train or test kind
        self.train = train

        # load data
        D = np.loadtxt(f'data/{"train" if self.train else "test"}.csv', delimiter=',', skiprows=1)

        # data to tensor
        D = torch.tensor(D)
        
        # if in train data set
        if self.train:
    
            # extract target labels
            self.y = D[:, 0].long()

        # create input tensor
        x = D[:, int(self.train):]

        # make the images quadratic (28 * 28 == 784)
        x = torch.reshape(x, (x.shape[0], 28, 28))

        # add empty dim for chanels
        x = x[:, None, :, :].float()

        # value range to [0, 1]
        self.x = x / 255
         
    # define length of dataset
    def __len__(self):
        
        # return sample count
        return self.x.shape[0]

    # define what dataset[idx] should mean
    def __getitem__(self, idx):

        # when training data
        if self.train:
        
            # return x and y
            return self.x[idx], self.y[idx]  

        # when in test return just x 
        return self.x[idx] 


# dev calls
def main():
    dataset = Dataset()
    for i in range(2):
        print(dataset[i])

if __name__ == '__main__':
    main()
