import torch
from torch import nn
from Functions import SLinearFunction, SMSEFunction, SCrossEntropyLossFunction

class SLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.op = nn.Linear(in_features, out_features, bias)
        self.weightS = nn.Parameter(torch.ones(self.op.weight.size()).requires_grad_())
        self.noise = torch.zeros_like(self.op.weight)
        self.function = SLinearFunction.apply
    
    def set_noise(self, var):
        self.noise = torch.normal(mean=0., std=var, size=self.noise.size()).to(self.op.weight.device)
    
    def clear_noise(self):
        self.noise = torch.zeros_like(self.op.weight)

    def push_S_device(self):
        self.weightS = self.weightS.to(self.op.weight.device)

    def clear_S_grad(self):
        with torch.no_grad():
            if self.weightS.grad is not None:
                self.weightS.grad.data *= 0
    
    def fetch_S_grad(self):
        return self.weightS.grad.sum()

    def do_second(self):
        self.op.weight.grad.data = self.op.weight.grad.data / (self.weightS.grad.data + 1e-10)

    def forward(self, x, xS):
        x, xS = self.function(x, xS, self.op.weight + self.noise, self.weightS, self.op.bias)
        return x, xS

class SReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
    
    def forward(self, x, xS):
        return self.relu(x), self.relu(xS)

class SModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = SLinear(28*28,32)
        self.fc2 = SLinear(32,32)
        self.fc3 = SLinear(32,10)
        self.relu = SReLU()
    
    def push_S_device(self):
        for m in self.modules():
            if isinstance(m, SLinear):
                m.push_S_device()

    def clear_S_grad(self):
        for m in self.modules():
            if isinstance(m, SLinear):
                m.clear_S_grad()

    def do_second(self):
        for m in self.modules():
            if isinstance(m, SLinear):
                m.do_second()

    def fetch_S_grad(self):
        S_grad_sum = 0
        for m in self.modules():
            if isinstance(m, SLinear):
                S_grad_sum += m.fetch_S_grad()
        return S_grad_sum
    
    def set_noise(self, var):
        for m in self.modules():
            if isinstance(m, SLinear):
                m.set_noise(var)
    
    def clear_noise(self):
        for m in self.modules():
            if isinstance(m, SLinear):
                m.clear_noise()


    def forward(self, x):
        xS = torch.zeros_like(x)
        x, xS = self.fc1(x, xS)
        x, xS = self.relu(x, xS)
        x, xS = self.fc2(x, xS)
        x, xS = self.relu(x, xS)
        x, xS = self.fc3(x, xS)
        return x, xS

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