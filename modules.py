import torch
from torch import nn
from Functions import SLinearFunction, SMSEFunction, SCrossEntropyLossFunction

class SLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.op = nn.Linear(in_features, out_features, bias)
        self.weightS = nn.Parameter(torch.ones(self.op.weight.size()).requires_grad_())
        self.noise = torch.zeros_like(self.op.weight)
        self.mask = torch.ones_like(self.op.weight)
        self.function = SLinearFunction.apply
    
    def set_noise(self, var):
        self.noise = torch.normal(mean=0., std=var, size=self.noise.size()).to(self.op.weight.device)
    
    def clear_noise(self):
        self.noise = torch.zeros_like(self.op.weight)
    
    def set_mask(self, portion):
        th = len(self.weightS.view(-1)) * (1-portion)
        self.mask = self.weightS.view(-1).sort()[1].view(self.weightS.size()) <= th
        # self.mask = (self.weightS.grad.data.abs() < portion).to(torch.float)
    
    def clear_mask(self):
        self.mask = torch.ones_like(self.op.weight)

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
        x, xS = self.function(x, xS, (self.op.weight + self.noise) * self.mask, self.weightS, self.op.bias)
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
    
    def set_mask(self, th):
        for m in self.modules():
            if isinstance(m, SLinear):
                m.set_mask(th)
    
    def clear_mask(self):
        for m in self.modules():
            if isinstance(m, SLinear):
                m.clear_mask()