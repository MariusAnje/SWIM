import torch
from torch import nn
from torch._C import device
from TFunctions import TLinearFunction, TConv2dFunction, TMSEFunction, TCrossEntropyLossFunction    

class TModule(nn.Module):
    def __init__(self):
        super().__init__()
    
    def create_helper(self):
        self.weightT = nn.Parameter(torch.ones(self.op.weight.size()).requires_grad_())
        self.noise = torch.zeros_like(self.op.weight)
        self.mask = torch.ones_like(self.op.weight)

    def set_noise(self, var):
        self.noise = torch.normal(mean=0., std=var, size=self.noise.size()).to(self.op.weight.device)
    
    def clear_noise(self):
        self.noise = torch.zeros_like(self.op.weight)
    
    def set_mask(self, portion, mode):
        if mode == "portion":
            th = self.weightT.grad.abs().view(-1).quantile(1-portion)
            self.mask = (self.weightT.grad.data.abs() <= th).to(torch.float)
        elif mode == "th":
            self.mask = (self.weightT.grad.data.abs() <= portion).to(torch.float)
        else:
            raise NotImplementedError(f"Mode: {mode} not supported, only support mode portion & th, ")
    
    def clear_mask(self):
        self.mask = torch.ones_like(self.op.weight)

    def push_T_device(self):
        self.weightT = self.weightT.to(self.op.weight.device)
        self.mask = self.mask.to(self.op.weight.device)

    def clear_T_grad(self):
        with torch.no_grad():
            if self.weightT.grad is not None:
                self.weightT.grad.data *= 0
    
    def fetch_T_grad(self):
        return (self.weightT.grad.abs() * self.mask).sum()
    
    def fetch_T_grad_list(self):
        return (self.weightT.grad.data * self.mask)

    def do_third(self, alpha):
        assert(alpha <= 1 and alpha >= 0)
        self.op.weight.grad.data = (1 - alpha) * self.op.weight.grad.data + alpha * self.weightT.grad.data

class TLinear(TModule):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.op = nn.Linear(in_features, out_features, bias)
        self.create_helper()
        self.function = TLinearFunction.apply

    def forward(self, x, xT):
        x, xT = self.function(x, xT, (self.op.weight + self.noise) * self.mask, self.weightT, self.op.bias)
        return x, xT

class TConv2d(TModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__()
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.create_helper()
        self.function = TConv2dFunction.apply

    def forward(self, x, xT):
        x, xT = self.function(x, xT, (self.op.weight + self.noise) * self.mask, self.weightT, self.op.bias, self.op.stride, self.op.padding, self.op.dilation, self.op.groups)
        return x, xT

class TReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.op = nn.ReLU()
    
    def forward(self, x, xT):
        return self.op(x), self.op(xT)

class TMaxpool2D(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super().__init__()
        return_indices = True
        self.op = nn.MaxPool2d(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
    
    def parse_indice(self, indice):
        bs, ch, w, h = indice.shape
        length = w * h
        BD = torch.LongTensor(list(range(bs))).expand([length,ch,bs]).swapaxes(0,2).reshape(-1)
        CD = torch.LongTensor(list(range(ch))).expand([bs,length,ch]).swapaxes(1,2).reshape(-1)
        shape = [bs, ch, -1]
        return shape, [BD, CD, indice.view(-1)]

    
    def forward(self, x, xT):
        x, indices = self.op(x)
        shape, indices = self.parse_indice(indices)
        xT = xT.view(shape)[indices].view(x.shape)
        return x, xT

class FakeTModule(nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op
        if isinstance(self.op, nn.MaxPool2d):
            self.op.return_indices = False
    
    def forward(self, x, xT):
        x = self.op(x)
        return x, None

class TModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def push_T_device(self):
        for m in self.modules():
            if isinstance(m, TLinear) or isinstance(m, TConv2d):
                m.push_T_device()

    def clear_T_grad(self):
        for m in self.modules():
            if isinstance(m, TLinear) or isinstance(m, TConv2d):
                m.clear_T_grad()

    def do_third(self, alpha):
        for m in self.modules():
            if isinstance(m, TLinear) or isinstance(m, TConv2d):
                m.do_third(alpha)

    def fetch_T_grad(self):
        T_grad_sum = 0
        for m in self.modules():
            if isinstance(m, TLinear) or isinstance(m, TConv2d):
                T_grad_sum += m.fetch_T_grad()
        return T_grad_sum
    
    def calc_T_grad_th(self, quantile):
        T_grad_list = None
        for m in self.modules():
            if isinstance(m, TLinear) or isinstance(m, TConv2d):
                if T_grad_list is None:
                    T_grad_list = m.fetch_T_grad_list().view(-1)
                else:
                    T_grad_list = torch.cat([T_grad_list, m.fetch_T_grad_list().view(-1)])
        th = torch.quantile(T_grad_list, 1-quantile)
        # print(th)
        return th

    def set_noise(self, var):
        for m in self.modules():
            if isinstance(m, TLinear) or isinstance(m, TConv2d):
                m.set_noise(var)
    
    def clear_noise(self):
        for m in self.modules():
            if isinstance(m, TLinear) or isinstance(m, TConv2d):
                m.clear_noise()
    
    def set_mask(self, th, mode):
        for m in self.modules():
            if isinstance(m, TLinear) or isinstance(m, TConv2d):
                m.set_mask(th, mode)
    
    def clear_mask(self):
        for m in self.modules():
            if isinstance(m, TLinear) or isinstance(m, TConv2d):
                m.clear_mask()
    
    def to_fake(self, device):
        for name, m in self.named_modules():
            if isinstance(m, TLinear) or isinstance(m, TConv2d) or isinstance(m, TMaxpool2D) or isinstance(m, TReLU):
                new = FakeTModule(m.op)
                self._modules[name] = new
        self.to(device)
    
    def back_real(self, device):
        for name, m in self.named_modules():
            if isinstance(m, FakeTModule):
                if isinstance(m.op, nn.Linear):
                    if m.op.bias is not None:
                        bias = True
                    new = TLinear(m.op.in_features, m.op.out_features, bias)
                    new.op = m.op
                    self._modules[name] = new

                elif isinstance(m.op, nn.Conv2d):
                    if m.op.bias is not None:
                        bias = True
                    new = TConv2d(m.op.in_channels, m.op.out_channels, m.op.kernel_size, m.op.stride, m.op.padding, m.op.dilation, m.op.groups, bias, m.op.padding_mode)
                    new.op = m.op
                    self._modules[name] = new

                elif isinstance(m.op, nn.MaxPool2d):
                    new = TMaxpool2D(m.op.kernel_size, m.op.stride, m.op.padding, m.op.dilation, m.op.return_indices, m.op.ceil_mode)
                    new.op = m.op
                    new.op.return_indices = True
                    self._modules[name] = new

                elif isinstance(m.op, nn.ReLU):
                    new = TReLU()
                    new.op = m.op
                    self._modules[name] = new

                else:
                    raise NotImplementedError
        self.to(device)