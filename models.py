import torch
from torch import nn
from Functions import SCrossEntropyLossFunction
from modules import SLinear, SReLU, SModel

class SCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super().__init__()
        self.function = SCrossEntropyLossFunction
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.reduce = reduce
        self.reduction = reduction
    
    def forward(self, input, inputS, labels):
        output = self.function.apply(input, inputS, labels, self.weight, self.size_average, self.ignore_index, self.reduce, self.reduction)
        return output

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28,32)
        self.fc2 = nn.Linear(32,32)
        self.fc3 = nn.Linear(32,10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SMLP3(SModel):
    def __init__(self):
        super().__init__()
        self.fc1 = SLinear(28*28,32)
        self.fc2 = SLinear(32,32)
        self.fc3 = SLinear(32,10)
        self.relu = SReLU()
    
    def forward(self, x):
        xS = torch.zeros_like(x)
        x, xS = self.fc1(x, xS)
        x, xS = self.relu(x, xS)
        x, xS = self.fc2(x, xS)
        x, xS = self.relu(x, xS)
        x, xS = self.fc3(x, xS)
        return x, xS

class SMLP4(SModel):
    def __init__(self):
        super().__init__()
        self.fc1 = SLinear(28*28,32)
        self.fc2 = SLinear(32,32)
        self.fc3 = SLinear(32,32)
        self.fc4 = SLinear(32,10)
        self.relu = SReLU()
    
    def forward(self, x):
        xS = torch.zeros_like(x)
        x, xS = self.fc1(x, xS)
        x, xS = self.relu(x, xS)
        x, xS = self.fc2(x, xS)
        x, xS = self.relu(x, xS)
        x, xS = self.fc3(x, xS)
        x, xS = self.relu(x, xS)
        x, xS = self.fc4(x, xS)
        return x, xS

