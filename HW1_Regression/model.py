import torch.nn as nn

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