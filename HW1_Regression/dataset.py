import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

data_root = '/home/zacharyyeh/Datasets/'
rescale = StandardScaler()

class COVID19Dataset(Dataset):
    def __init__(self, mode = "train"):
        self.mode = mode
        #(2700, 94)
        self.data = np.loadtxt(data_root+'COVID19/covid.train.csv', delimiter = ",", dtype = np.float32, skiprows = 1)[:, 1:]
        #standardize data, started at 40 column(1-39 is one-hot encoded)
        rescale.fit(self.data[:, 40:])
        self.data[:, 40:] = rescale.transform(self.data[:, 40:])
        if self.mode == "train":
            #(2001, 94)
            self.data = self.data[:2000, :]
        elif self.mode == "test":
            #(699, 94)
            self.data = self.data[2001:, :] 
        #Note!!: Array[:, 0:b] only return b columns not b+1
        #So specify b as the number you want for your columns
        #self.data[:, 0:93] takes column 0 to 92!!! 93 columns in total
        self.features = torch.from_numpy(self.data[:, 0:93]).to(torch.long)
        self.labels = torch.from_numpy(self.data[:, 93])

    def __getitem__(self, index):
        #label[index] return a floating point, we make it a tensor [float]
        return self.features[index, :], self.labels[index].reshape(1)
    
    def __len__(self):
        return self.data.shape[0]

class COVID19DRDataset(Dataset):
    def __init__(self, mode = "train"):
        self.mode = mode
        #(2700, 94)
        self.map = np.zeros(40)
        self.count = np.zeros(40)
        self.data = np.loadtxt(data_root+'COVID19/covid.train.csv', delimiter = ",", dtype = np.float32, skiprows = 1)[:, 1:]

        for i in range(self.data.shape[0]):
            for j in range(40):
                if self.data[i, j] == 1:
                    self.map[j] += self.data[i, 93]
                    self.count[j]  += 1

        self.map = self.map / self.count
        
        for i in range(self.data.shape[0]):
            for j in range(40):
                if self.data[i, j] == 1:
                    self.data[i, 39] = self.map[j]
        self.data = self.data[:, 39:]
        #standardize data, started at 40 column(1-39 is one-hot encoded)
        rescale.fit(self.data[:, :])
        self.data[:, :] = rescale.transform(self.data[:, :])
        if self.mode == "train":
            #(2001, )
            self.data = self.data[:2000, :]
        elif self.mode == "test":
            #(699, )
            self.data = self.data[2001:, :] 
        #Note!!: Array[:, 0:b] only return b columns not b+1
        #So specify b as the number you want for your columns
        #self.data[:, 0:93] takes column 0 to 92!!! 93 columns in total
        self.features = torch.from_numpy(self.data[:, 0:54])
        self.labels = torch.from_numpy(self.data[:, -1])


    def __getitem__(self, index):
        #label[index] return a floating point, we make it a tensor [float]
        return self.features[index, :], self.labels[index].reshape(1)
    
    def __len__(self):
        return self.data.shape[0]
