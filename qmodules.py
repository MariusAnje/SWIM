import torch
from modules import SModule, NModule
from Functions import QuantFunction
from torch import nn
from Functions import SLinearFunction, SConv2dFunction, SMSEFunction, SCrossEntropyLossFunction, SBatchNorm2dFunction

quant = QuantFunction.apply

class QSLinear(SModule):
    def __init__(self, N, in_features, out_features, bias=True):
        super().__init__()
        self.op = nn.Linear(in_features, out_features, bias)
        self.create_helper()
        self.function = SLinearFunction.apply
        self.N = N
    
    def copy_N(self):
        new = QNLinear(self.N, self.op.in_features, self.op.out_features, False if self.op.bias is None else True)
        new.op = self.op
        new.noise = self.noise
        new.mask = self.mask
        return new

    def forward(self, xC):
        x, xS = xC
        x, xS = self.function(x * self.scale, xS * self.scale, quant(self.N, self.op.weight) + self.noise, self.weightS)
        if self.op.bias is not None:
            x += quant(self.N, self.op.bias)
        if self.op.bias is not None:
            xS += self.op.bias
        return quant(self.N, x), xS

class QSConv2d(SModule):
    def __init__(self, N, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__()
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.create_helper()
        self.function = SConv2dFunction.apply
        self.N = N

    def copy_N(self):
        new = QNConv2d(self.N, self.op.in_channels, self.op.out_channels, self.op.kernel_size, self.op.stride, self.op.padding, self.op.dilation, self.op.groups, False if self.op.bias is None else True, self.op.padding_mode)
        new.op = self.op
        new.noise = self.noise
        new.mask = self.mask
        return new

    def forward(self, xC):
        x, xS = xC
        x, xS = self.function(x * self.scale, xS * self.scale, quant(self.N, self.op.weight) + self.noise, self.weightS, None, self.op.stride, self.op.padding, self.op.dilation, self.op.groups)
        if self.op.bias is not None:
            x += quant(self.N, self.op.bias).reshape(1,-1,1,1).expand_as(x)
        if self.op.bias is not None:
            xS += self.op.bias.reshape(1,-1,1,1).expand_as(xS)
        return quant(self.N, x), xS

class QNLinear(NModule):
    def __init__(self, N, in_features, out_features, bias=True):
        super().__init__()
        self.op = nn.Linear(in_features, out_features, bias)
        self.noise = torch.zeros_like(self.op.weight)
        self.mask = torch.ones_like(self.op.weight)
        self.function = nn.functional.linear
        self.N = N

    def copy_S(self):
        new = QSLinear(self.N, self.op.in_features, self.op.out_features, False if self.op.bias is None else True)
        new.op = self.op
        new.noise = self.noise
        new.mask = self.mask
        return new

    def forward(self, x):
        x = self.function(x, quant(self.N, self.op.weight) + self.noise, self.op.bias)
        return quant(self.N, x)

class QNConv2d(NModule):
    def __init__(self, N, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__()
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.noise = torch.zeros_like(self.op.weight)
        self.mask = torch.ones_like(self.op.weight)
        self.function = nn.functional.conv2d
        self.N = N
    
    def copy_S(self):
        new = QSConv2d(self.N, self.op.in_channels, self.op.out_channels, self.op.kernel_size, self.op.stride, self.op.padding, self.op.dilation, self.op.groups, False if self.op.bias is None else True, self.op.padding_mode)
        new.op = self.op
        new.noise = self.noise
        new.mask = self.mask
        return new

    def forward(self, x):
        x = self.function(x, quant(self.N, self.op.weight) + self.noise, self.op.bias, self.op.stride, self.op.padding, self.op.dilation, self.op.groups)
        return quant(self.N, x)
