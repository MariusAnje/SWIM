import torch
from torch import nn
from Functions import SCrossEntropyLossFunction
from modules import SLinear, SReLU, SModel, SConv2d, SMaxpool2D


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

class FakeSCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super().__init__()
        self.op = nn.CrossEntropyLoss(weight, size_average, ignore_index, reduce, reduction)
    
    def forward(self, input, inputS, labels):
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

class SMLP3(SModel):
    def __init__(self):
        super().__init__()
        self.fc1 = SLinear(28*28,32)
        self.fc2 = SLinear(32,32)
        self.fc3 = SLinear(32,10)
        self.relu = SReLU()
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        xS = torch.zeros_like(x)
        x = self.fc1((x, xS))
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

class SMLP4(SModel):
    def __init__(self):
        super().__init__()
        self.fc1 = SLinear(28*28,32)
        self.fc2 = SLinear(32,32)
        self.fc3 = SLinear(32,32)
        self.fc4 = SLinear(32,10)
        self.relu = SReLU()
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        xS = torch.zeros_like(x)
        x = self.fc1((x, xS))
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x


class SLeNet(SModel):

    def __init__(self):
        super().__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = SConv2d(1, 6, 3, padding=1)
        self.conv2 = SConv2d(6, 16, 3, padding=1)
        # an affine operation: y = Wx + b
        self.fc1 = SLinear(16 * 7 * 7, 120)  # 6*6 from image dimension
        self.fc2 = SLinear(120, 84)
        self.fc3 = SLinear(84, 10)
        self.pool = SMaxpool2D(2)
        self.relu = SReLU()

    def forward(self, x):
        xS = torch.zeros_like(x)
        x = self.conv1((x, xS))
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x, xS = x
        x = x.view(-1, self.num_flat_features(x))
        if xS is not None:
            xS = xS.view(-1, self.num_flat_features(xS))
        
        x = self.fc1((x, xS))
        x = self.relu(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class CIFAR(SModel):
    def __init__(self):
        super().__init__()

        self.conv1 = SConv2d(3, 64, 3, padding=1)
        self.conv2 = SConv2d(64, 64, 3, padding=1)
        self.pool1 = SMaxpool2D(2,2)

        self.conv3 = SConv2d(64,128,3, padding=1)
        self.conv4 = SConv2d(128,128,3, padding=1)
        self.pool2 = SMaxpool2D(2,2)

        self.conv5 = SConv2d(128,256,3, padding=1)
        self.conv6 = SConv2d(256,256,3, padding=1)
        self.pool3 = SMaxpool2D(2,2)
        
        self.fc1 = SLinear(256 * 4 * 4, 1024)
        self.fc2 = SLinear(1024, 1024)
        self.fc3 = SLinear(1024, 10)
        self.relu = SReLU()

    def forward(self, x):
        xS = torch.zeros_like(x)
        x = self.conv1((x, xS))
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.pool3(x)
        
        x, xS = x
        x = x.view(-1, self.num_flat_features(x))
        if xS is not None:
            xS = xS.view(-1, self.num_flat_features(xS))
        
        x = self.fc1((x, xS))
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

from modules import NConv2d, NLinear
class FakeCIFAR(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = NConv2d(3, 64, 3, padding=1)
        self.conv2 = NConv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2,2)

        self.conv3 = NConv2d(64,128,3, padding=1)
        self.conv4 = NConv2d(128,128,3, padding=1)
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv5 = NConv2d(128,256,3, padding=1)
        self.conv6 = NConv2d(256,256,3, padding=1)
        self.pool3 = nn.MaxPool2d(2,2)
        
        self.fc1 = NLinear(256 * 4 * 4, 1024)
        self.fc2 = NLinear(1024, 1024)
        self.fc3 = NLinear(1024, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.pool3(x)
        
        x = x.view(-1, self.num_flat_features(x))
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    def set_noise(self, var, N=8, m=1):
        for mo in self.modules():
            if isinstance(mo, NLinear) or isinstance(mo, NConv2d):
                # m.set_noise(var)
                mo.set_noise(var, N, m)
    
    def clear_noise(self):
        for m in self.modules():
            if isinstance(m, NLinear) or isinstance(m, NConv2d):
                m.clear_noise()
