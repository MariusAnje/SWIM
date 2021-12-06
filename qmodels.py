import torch
from torch import nn
from Functions import SCrossEntropyLossFunction
from modules import SReLU, SModel, SMaxpool2D
from qmodules import QSLinear, QSConv2d

class QSMLP3(SModel):
    def __init__(self, N=4):
        super().__init__()
        self.fc1 = QSLinear(N, 28*28,32)
        self.fc2 = QSLinear(N, 32,32)
        self.fc3 = QSLinear(N, 32,10)
        self.relu = SReLU()
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        xS = torch.zeros_like(x)
        if not self.first_only:
            x = (x, xS)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

class QSMLP4(SModel):
    def __init__(self, N=4):
        super().__init__()
        self.fc1 = QSLinear(N, 28*28,32)
        self.fc2 = QSLinear(N, 32,32)
        self.fc3 = QSLinear(N, 32,32)
        self.fc4 = QSLinear(N, 32,10)
        self.relu = SReLU()
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        xS = torch.zeros_like(x)
        if not self.first_only:
            x = (x, xS)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x


class QSLeNet(SModel):

    def __init__(self, N=4):
        super().__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = QSConv2d(N, 1, 6, 3, padding=1)
        self.conv2 = QSConv2d(N, 6, 16, 3, padding=1)
        # an affine operation: y = Wx + b
        self.fc1 = QSLinear(N, 16 * 7 * 7, 120)  # 6*6 from image dimension
        self.fc2 = QSLinear(N, 120, 84)
        self.fc3 = QSLinear(N, 84, 10)
        self.pool = SMaxpool2D(2)
        self.relu = SReLU()

    def forward(self, x):
        xS = torch.zeros_like(x)
        if not self.first_only:
            x = (x, xS)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.unpack_flattern(x)
        
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

class QCIFAR(SModel):
    def __init__(self, N=6):
        super().__init__()

        self.conv1 = QSConv2d(N, 3, 64, 3, padding=1)
        self.conv2 = QSConv2d(N, 64, 64, 3, padding=1)
        self.pool1 = SMaxpool2D(2,2)

        self.conv3 = QSConv2d(N, 64,128,3, padding=1)
        self.conv4 = QSConv2d(N, 128,128,3, padding=1)
        self.pool2 = SMaxpool2D(2,2)

        self.conv5 = QSConv2d(N, 128,256,3, padding=1)
        self.conv6 = QSConv2d(N, 256,256,3, padding=1)
        self.pool3 = SMaxpool2D(2,2)
        
        self.fc1 = QSLinear(N, 256 * 4 * 4, 1024)
        self.fc2 = QSLinear(N, 1024, 1024)
        self.fc3 = QSLinear(N, 1024, 10)
        self.relu = SReLU()

    def forward(self, x):
        xS = torch.zeros_like(x)
        if not self.first_only:
            x = (x, xS)
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
        
        x = self.unpack_flattern(x)
 
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
