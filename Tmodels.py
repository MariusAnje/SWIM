import torch
from torch import nn
from TFunctions import TCrossEntropyLossFunction
from Tmodules import TLinear, TReLU, TModel, TConv2d, TMaxpool2D


class TCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super().__init__()
        self.function = TCrossEntropyLossFunction
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.reduce = reduce
        self.reduction = reduction
    
    def forward(self, input, inputT, labels):
        output = self.function.apply(input, inputT, labels, self.weight, self.size_average, self.ignore_index, self.reduce, self.reduction)
        return output

class FakeTCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super().__init__()
        self.op = nn.CrossEntropyLoss(weight, size_average, ignore_index, reduce, reduction)
    
    def forward(self, input, inputT, labels):
        output = self.op(input, labels)
        return output

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28,32)
        self.fc2 = nn.Linear(32,32)
        self.fc3 = nn.Linear(32,10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class TMLP3(TModel):
    def __init__(self):
        super().__init__()
        self.fc1 = TLinear(28*28,32)
        self.fc2 = TLinear(32,32)
        self.fc3 = TLinear(32,10)
        self.relu = TReLU()
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        xT = torch.zeros_like(x)
        x, xT = self.fc1(x, xT)
        x, xT = self.relu(x, xT)
        x, xT = self.fc2(x, xT)
        x, xT = self.relu(x, xT)
        x, xT = self.fc3(x, xT)
        return x, xT

class TMLP4(TModel):
    def __init__(self):
        super().__init__()
        self.fc1 = TLinear(28*28,32)
        self.fc2 = TLinear(32,32)
        self.fc3 = TLinear(32,32)
        self.fc4 = TLinear(32,10)
        self.relu = TReLU()
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        xT = torch.zeros_like(x)
        x, xT = self.fc1(x, xT)
        x, xT = self.relu(x, xT)
        x, xT = self.fc2(x, xT)
        x, xT = self.relu(x, xT)
        x, xT = self.fc3(x, xT)
        x, xT = self.relu(x, xT)
        x, xT = self.fc4(x, xT)
        return x, xT


class TLeNet(TModel):

    def __init__(self):
        super().__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = TConv2d(1, 6, 3, padding=1)
        self.conv2 = TConv2d(6, 16, 3, padding=1)
        # an affine operation: y = Wx + b
        self.fc1 = TLinear(16 * 7 * 7, 120)  # 6*6 from image dimension
        self.fc2 = TLinear(120, 84)
        self.fc3 = TLinear(84, 10)
        self.pool = TMaxpool2D(2)
        self.relu = TReLU()

    def forward(self, x):
        xT = torch.zeros_like(x)
        x, xT = self.conv1(x, xT)
        x, xT = self.relu(x, xT)
        x, xT = self.pool(x, xT)
        
        x, xT = self.conv2(x, xT)
        x, xT = self.relu(x, xT)
        x, xT = self.pool(x, xT)
        
        x = x.view(-1, self.num_flat_features(x))
        if xT is not None:
            xT = xT.view(-1, self.num_flat_features(xT))
        
        x, xT = self.fc1(x, xT)
        x, xT = self.relu(x, xT)
        
        x, xT = self.fc2(x, xT)
        x, xT = self.relu(x, xT)
        
        x, xT = self.fc3(x, xT)
        return x, xT

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
