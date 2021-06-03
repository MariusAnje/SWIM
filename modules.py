import torch
from torch import nn
from Functions import SLinearFunction, SConv2dFunction, SMSEFunction, SCrossEntropyLossFunction

class SModule(nn.Module):
    def __init__(self):
        super().__init__()
    
    def create_helper(self):
        self.weightS = nn.Parameter(torch.ones(self.op.weight.size()).requires_grad_())
        self.noise = torch.zeros_like(self.op.weight)
        self.mask = torch.ones_like(self.op.weight)

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

class SLinear(SModule):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.op = nn.Linear(in_features, out_features, bias)
        self.create_helper()
        self.function = SLinearFunction.apply

    def forward(self, x, xS):
        x, xS = self.function(x, xS, (self.op.weight + self.noise) * self.mask, self.weightS, self.op.bias)
        return x, xS

class SConv2d(SModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__()
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.create_helper()
        self.function = SConv2dFunction.apply

    def forward(self, x, xS):
        x, xS = self.function(x, xS, (self.op.weight + self.noise) * self.mask, self.weightS, self.op.bias, self.op.padding, self.op.dilation, self.op.groups)
        return x, xS

class SReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
    
    def forward(self, x, xS):
        return self.relu(x), self.relu(xS)

class SMaxpool2D(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super().__init__()
        return_indices = True
        self.pool = nn.MaxPool2d(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
    
    def parse_indice(self, indice):
        bs, ch, w, h = indice.shape
        length = w * h
        BD = torch.LongTensor(list(range(bs))).expand([length,ch,bs]).swapaxes(0,2).reshape(-1)
        CD = torch.LongTensor(list(range(ch))).expand([bs,length,ch]).swapaxes(1,2).reshape(-1)
        shape = [bs, ch, -1]
        return shape, [BD, CD, indice.view(-1)]

    
    def forward(self, x, xS):
        x, indices = self.pool(x)
        shape, indices = self.parse_indice(indices)
        xS = xS.view(shape)[indices].view(x.shape)
        return x, xS


class SModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def push_S_device(self):
        for m in self.modules():
            if isinstance(m, SLinear) or isinstance(m, SConv2d):
                m.push_S_device()

    def clear_S_grad(self):
        for m in self.modules():
            if isinstance(m, SLinear) or isinstance(m, SConv2d):
                m.clear_S_grad()

    def do_second(self):
        for m in self.modules():
            if isinstance(m, SLinear) or isinstance(m, SConv2d):
                m.do_second()

    def fetch_S_grad(self):
        S_grad_sum = 0
        for m in self.modules():
            if isinstance(m, SLinear) or isinstance(m, SConv2d):
                S_grad_sum += m.fetch_S_grad()
        return S_grad_sum
    
    def set_noise(self, var):
        for m in self.modules():
            if isinstance(m, SLinear) or isinstance(m, SConv2d):
                m.set_noise(var)
    
    def clear_noise(self):
        for m in self.modules():
            if isinstance(m, SLinear) or isinstance(m, SConv2d):
                m.clear_noise()
    
    def set_mask(self, th):
        for m in self.modules():
            if isinstance(m, SLinear) or isinstance(m, SConv2d):
                m.set_mask(th)
    
    def clear_mask(self):
        for m in self.modules():
            if isinstance(m, SLinear) or isinstance(m, SConv2d):
                m.clear_mask()