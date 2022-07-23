import torch.nn as nn
import torch
import numpy as np

class Regression(nn.Module):
    def __init__(self):
        super(Regression, self).__init__()
        #input features = 93, output features = 1
        self.net = nn.Sequential(
            nn.Linear(93, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)
         
class DRRegression(nn.Module):
    def __init__(self):
        super(DRRegression, self).__init__()
        #input features = 93, output features = 1
        self.net = nn.Sequential(
            nn.Linear(54, 27),
            nn.ReLU(),
            nn.Linear(27, 1)
        )

    def forward(self, x):
        return self.net(x)

class EmbeddedRegression(nn.Module):
    def __init__(self):
        super(EmbeddedRegression, self).__init__()
        #input features = 93, output features = 1
        self.embedded = nn.Embedding(16, 1)
        self.net = nn.Sequential(
            nn.Linear(93, 27),
            nn.ReLU(),
            nn.Linear(27, 1)
        )
    def forward(self, x):
        self.onehot, self.features = torch.tensor_split(x, torch.tensor([40]).to(torch.long), dim=1)
        self.onehot = self.embedded(self.onehot)
        out = torch.cat((torch.squeeze(self.onehot), self.features), dim=1)
        out = self.net(out)
        return out