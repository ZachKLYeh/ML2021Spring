import numpy as np
import torch
from torch.utils.data import Dataset
import math
from sklearn.preprocessing import StandardScaler

rescale = StandardScaler()

data_root='/home/zacharyyeh/Datasets/timit_11/'
#first specify numpy datatype, this will be inheret to tensor dtype
train_data = np.load(data_root + 'train_11.npy').astype(np.float32)
train_label = np.load(data_root + 'train_label_11.npy').astype(np.long)
test_data = np.load(data_root + 'test_11.npy')

#normalize data and label
train_data = rescale.fit_transform(train_data)

#specify train/val/test split
datacount = math.floor(train_data.shape[0])
#train/val split the rest is test
split = [0.8, 0.1]
train_count = math.floor(datacount*split[0])
val_count = math.floor(train_count+datacount*split[1])

class TIMITDataset(Dataset):
    def __init__(self, mode = "train"):
        self.mode = mode
        self.train_data = torch.from_numpy(train_data)
        self.train_label = torch.from_numpy(train_label)
        if self.mode == "train":
            self.data = train_data[0:train_count, :]
            self.label = train_label[0:train_count]
        elif self.mode == "val":
            self.data = train_data[train_count:val_count, ]
            self.label = train_label[train_count:val_count]
        elif self.mode == "test":
            self.data = train_data[val_count:, ]
            self.label = train_label[val_count:]

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.data.shape[0]

        
        
