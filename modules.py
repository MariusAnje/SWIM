import torch
from torch import nn
from torch import functional
from torch._C import device
from torch.nn.modules.pooling import MaxPool2d
from Functions import SLinearFunction, SConv2dFunction, SMSEFunction, SCrossEntropyLossFunction, SBatchNorm2dFunction
import numpy as np

class SModule(nn.Module):
    def __init__(self):
        super().__init__()
    
    def create_helper(self):
        self.weightS = nn.Parameter(torch.ones(self.op.weight.size()).requires_grad_())
        self.noise = torch.zeros_like(self.op.weight)
        self.mask = torch.ones_like(self.op.weight)
        self.original_w = None
        self.original_b = None
        self.scale = 1.0
    
    def set_noise(self, dev_var, write_var, N, m):
        # N: number of bits per weight, m: number of bits per device
        # Dev_var: device variation before write and verify
        # write_var: device variation after write and verity
        scale = self.op.weight.abs().max()
        noise_dev = torch.zeros_like(self.noise).to(self.op.weight.device)
        noise_write = torch.zeros_like(self.noise).to(self.op.weight.device)
        for i in range(1, N//m + 1):
            if dev_var != 0:
                noise_dev   += (torch.normal(mean=0., std=dev_var, size=self.noise.size()) * (pow(2, - i*m))).to(self.op.weight.device)
            if write_var != 0:
                noise_write += (torch.normal(mean=0., std=write_var, size=self.noise.size()) * (pow(2, - i*m))).to(self.op.weight.device)
        noise_dev = noise_dev.to(self.op.weight.device) * scale
        noise_write = noise_write.to(self.op.weight.device) * scale

        self.noise = noise_dev * self.mask + noise_write * (1 - self.mask)
    
    def clear_noise(self):
        self.noise = torch.zeros_like(self.op.weight)
    
    def mask_indicator(self, method, alpha=None):
        if method == "second":
            return self.weightS.grad.data.abs()
        if method == "magnitude":
            return 1 / (self.op.weight.data.abs() + 1e-8)
        if method == "saliency":
            if alpha is None:
                alpha = 2
            return self.weightS.grad.data.abs() * (self.op.weight.data ** alpha).abs()
        if method == "r_saliency":
            if alpha is None:
                alpha = 2
            return self.weightS.grad.abs() * self.op.weight.abs().max() / (self.op.weight.data ** alpha + 1e-8).abs()
        if method == "subtract":
            return self.weightS.grad.data.abs() - alpha * self.weightS.grad.data.abs() * (self.op.weight.data ** 2)
        if method == "SM":
            return self.weightS.grad.data.abs() * alpha - self.op.weight.data.abs()
        else:
            raise NotImplementedError(f"method {method} not supported")
    
    def get_mask_info(self):
        total = (self.mask != 10).sum()
        RM = (self.mask == 0).sum()
        return total, RM

    def set_mask(self, portion, mode):
        if mode == "portion":
            th = self.weightS.grad.abs().view(-1).quantile(1-portion)
            mask = (self.weightS.grad.data.abs() <= th).to(torch.float)
        elif mode == "th":
            mask = (self.weightS.grad.data.abs() <= portion).to(torch.float)
        else:
            raise NotImplementedError(f"Mode: {mode} not supported, only support mode portion & th, ")
        self.mask = self.mask * mask
    
    def set_mask_mag(self, portion, mode):
        if mode == "portion":
            th = self.op.weight.abs().view(-1).quantile(1-portion)
            self.mask = (self.op.weight.data.abs() <= th).to(torch.float)
        elif mode == "th":
            self.mask = (self.op.weight.data.abs() <= portion).to(torch.float)
        else:
            raise NotImplementedError(f"Mode: {mode} not supported, only support mode portion & th, ")
    
    def set_mask_sail(self, portion, mode, method, alpha=None):
        if mode == "random":
            size = len(self.mask.view(-1))
            self.mask = torch.Tensor(np.random.binomial(1,1-portion,size)).to(self.mask.dtype).to(self.mask.device).view(self.mask.shape)
        else:
            saliency = self.mask_indicator(method, alpha)
            if mode == "portion":
                th = saliency.view(-1).quantile(1-portion)
                mask = (saliency <= th).to(torch.float)
            elif mode == "th":
                mask = (saliency <= portion).to(torch.float)                
            else:
                raise NotImplementedError(f"Mode: {mode} not supported, only support mode portion & th, ")
            self.mask = self.mask * mask
    
    def clear_mask(self):
        self.mask = torch.ones_like(self.op.weight)

    def push_S_device(self):
        self.weightS = self.weightS.to(self.op.weight.device)
        self.mask = self.mask.to(self.op.weight.device)
        self.noise = self.noise.to(self.op.weight.device)

    def clear_S_grad(self):
        with torch.no_grad():
            if self.weightS.grad is not None:
                self.weightS.grad.data *= 0
    
    def fetch_S_grad(self):
        return (self.weightS.grad.abs() * self.mask).sum()
    
    def fetch_S_grad_list(self):
        return (self.weightS.grad.data * self.mask)

    def do_second(self):
        self.op.weight.grad.data = self.op.weight.grad.data / (self.weightS.grad.data + 1e-10)
    
    def normalize(self):
        if self.original_w is None:
            self.original_w = self.op.weight.data
        if (self.original_b is None) and (self.op.bias is not None):
            self.original_b = self.op.bias.data
        scale = self.op.weight.data.abs().max().item()
        self.scale = scale
        self.op.weight.data = self.op.weight.data / scale
        # if self.op.bias is not None:
        #     self.op.bias.data = self.op.bias.data / scale

class SLinear(SModule):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.op = nn.Linear(in_features, out_features, bias)
        self.create_helper()
        self.function = SLinearFunction.apply
    
    def copy_N(self):
        new = NLinear(self.op.in_features, self.op.out_features, False if self.op.bias is None else True)
        new.op = self.op
        new.noise = self.noise
        new.mask = self.mask
        return new

    def forward(self, xC):
        x, xS = xC
        x, xS = self.function(x * self.scale, xS * self.scale, self.op.weight + self.noise, self.weightS)
        if self.op.bias is not None:
            x += self.op.bias
        if self.op.bias is not None:
            xS += self.op.bias
        return x, xS

class SConv2d(SModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__()
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.create_helper()
        self.function = SConv2dFunction.apply

    def copy_N(self):
        new = NConv2d(self.op.in_channels, self.op.out_channels, self.op.kernel_size, self.op.stride, self.op.padding, self.op.dilation, self.op.groups, False if self.op.bias is None else True, self.op.padding_mode)
        new.op = self.op
        new.noise = self.noise
        new.mask = self.mask
        return new

    def forward(self, xC):
        x, xS = xC
        x, xS = self.function(x * self.scale, xS * self.scale, self.op.weight + self.noise, self.weightS, None, self.op.stride, self.op.padding, self.op.dilation, self.op.groups)
        if self.op.bias is not None:
            x += self.op.bias.reshape(1,-1,1,1).expand_as(x)
        if self.op.bias is not None:
            xS += self.op.bias.reshape(1,-1,1,1).expand_as(xS)
        return x, xS

class NModule(nn.Module):
    def set_noise(self, dev_var, write_var, N, m):
        # N: number of bits per weight, m: number of bits per device
        # Dev_var: device variation before write and verify
        # write_var: device variation after write and verity
        scale = self.op.weight.abs().max()
        noise_dev = torch.zeros_like(self.noise).to(self.op.weight.device)
        noise_write = torch.zeros_like(self.noise).to(self.op.weight.device)
        for i in range(1, N//m + 1):
            if dev_var != 0:
                noise_dev   += (torch.normal(mean=0., std=dev_var, size=self.noise.size()) * (pow(2, - i*m))).to(self.op.weight.device)
            if write_var != 0:
                noise_write += (torch.normal(mean=0., std=write_var, size=self.noise.size()) * (pow(2, - i*m))).to(self.op.weight.device)
        noise_dev = noise_dev.to(self.op.weight.device) * scale
        noise_write = noise_write.to(self.op.weight.device) * scale

        self.noise = noise_dev * self.mask + noise_write * (1 - self.mask)
    
    def clear_noise(self):
        self.noise = torch.zeros_like(self.op.weight)
    
    def push_S_device(self):
        self.mask = self.mask.to(self.op.weight.device)
        self.noise = self.noise.to(self.op.weight.device)

class NLinear(NModule):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.op = nn.Linear(in_features, out_features, bias)
        self.noise = torch.zeros_like(self.op.weight)
        self.mask = torch.ones_like(self.op.weight)
        self.function = nn.functional.linear

    def copy_S(self):
        new = SLinear(self.op.in_features, self.op.out_features, False if self.op.bias is None else True)
        new.op = self.op
        new.noise = self.noise
        new.mask = self.mask
        return new

    def forward(self, x):
        x = self.function(x, self.op.weight + self.noise, self.op.bias)
        return x

class NConv2d(NModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__()
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.noise = torch.zeros_like(self.op.weight)
        self.mask = torch.ones_like(self.op.weight)
        self.function = nn.functional.conv2d
    
    def copy_S(self):
        new = SConv2d(self.op.in_channels, self.op.out_channels, self.op.kernel_size, self.op.stride, self.op.padding, self.op.dilation, self.op.groups, False if self.op.bias is None else True, self.op.padding_mode)
        new.op = self.op
        new.noise = self.noise
        new.mask = self.mask
        return new

    def forward(self, x):
        x = self.function(x, self.op.weight + self.noise, self.op.bias, self.op.stride, self.op.padding, self.op.dilation, self.op.groups)
        return x

class SReLU(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.op = nn.ReLU(inplace)
    
    def forward(self, xC):
        x, xS = xC
        with torch.no_grad():
            mask = (x > 0).to(torch.float)
        return self.op(x), xS * mask

class SMaxpool2D(nn.Module):
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

    
    def forward(self, xC):
        x, xS = xC
        x, indices = self.op(x)
        shape, indices = self.parse_indice(indices)
        xS = xS.view(shape)[indices].view(x.shape)
        return x, xS

class SAdaptiveAvgPool2d(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.op = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, xC):
        x, xS = xC
        return self.op(x), self.op(xS)

class SBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.op = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.function = SBatchNorm2dFunction.apply
    
    def forward(self, xC):
        x, xS = xC
        x, xS = self.function(x, xS, self.op.running_mean, self.op.running_var, self.op.weight, self.op.bias, self.op.training, self.op.momentum, self.op.eps)
        return x, xS

class FakeSModule(nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op
        if isinstance(self.op, nn.MaxPool2d):
            self.op.return_indices = False
    
    def forward(self, xC):
        x, xS = xC
        x = self.op(x)
        return x, None

class NModel(nn.Module):
    def __init__(self):
        super().__init__()

    def set_noise(self, dev_var, write_var, N=8, m=1):
        for mo in self.modules():
            if isinstance(mo, NModule):
                mo.set_noise(dev_var, write_var, N, m)
    
    def clear_noise(self):
        for m in self.modules():
            if isinstance(m, NModule):
                m.clear_noise()
    
    def push_S_device(self):
        for m in self.modules():
            if isinstance(m, NModule):
            # if isinstance(m, NLinear) or isinstance(m, NConv2d):
                m.push_S_device()

class SModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_only = False
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def unpack_flattern(self, x):
        if self.first_only:
            return x.view(-1, self.num_flat_features(x))
        else:
            x, xS = x
            x = x.view(-1, self.num_flat_features(x))
            if xS is not None:
                xS = xS.view(-1, self.num_flat_features(xS))
            return x, xS

    def push_S_device(self):
        for m in self.modules():
            if isinstance(m, SModule):
                m.push_S_device()

    def clear_S_grad(self):
        for m in self.modules():
            if isinstance(m, SModule):
                m.clear_S_grad()

    def do_second(self):
        for m in self.modules():
            if isinstance(m, SModule):
                m.do_second()

    def fetch_S_grad(self):
        S_grad_sum = 0
        for m in self.modules():
            if isinstance(m, SModule):
                S_grad_sum += m.fetch_S_grad()
        return S_grad_sum
    
    def calc_S_grad_th(self, quantile):
        S_grad_list = None
        for m in self.modules():
            if isinstance(m, SModule):
                if S_grad_list is None:
                    S_grad_list = m.fetch_S_grad_list().view(-1)
                else:
                    S_grad_list = torch.cat([S_grad_list, m.fetch_S_grad_list().view(-1)])
        th = torch.quantile(S_grad_list, 1-quantile)
        # print(th)
        return th
    
    def calc_sail_th(self, quantile, method, alpha=None):
        sail_list = None
        for m in self.modules():
            if isinstance(m, SModule):
                sail = m.mask_indicator(method, alpha).view(-1)
                if sail_list is None:
                    sail_list = sail
                else:
                    sail_list = torch.cat([sail_list, sail])
        # import time
        # torch.save(sail_list, f"S_grad_{time.time()}.pt")
        th = torch.quantile(sail_list, 1-quantile)
        # print(th)
        return th

    def set_noise(self, dev_var, write_var, N=8, m=1):
        for mo in self.modules():
            if isinstance(mo, SModule) or isinstance(mo, NModule):
                mo.set_noise(dev_var, write_var, N, m)
    
    def clear_noise(self):
        for m in self.modules():
            if isinstance(m, SModule):
                m.clear_noise()
    
    def get_mask_info(self):
        total = 0
        RM = 0
        for m in self.modules():
            if isinstance(m, SModule):
                t, r = m.get_mask_info()
                total += t
                RM += r
        return total, RM

    def set_mask(self, th, mode):
        for m in self.modules():
            if isinstance(m, SModule):
                m.set_mask(th, mode)
    
    def set_mask_mag(self, th, mode):
        for m in self.modules():
            if isinstance(m, SModule):
                m.set_mask_mag(th, mode)
    
    def set_mask_sail(self, th, mode, method, alpha=None):
        for m in self.modules():
            if isinstance(m, SModule):
                m.set_mask_sail(th, mode, method, alpha)
    
    def clear_mask(self):
        for m in self.modules():
            if isinstance(m, SModule):
                m.clear_mask()
    
    def to_fake(self, device):
        for name, m in self.named_modules():
            if isinstance(m, SModule) or isinstance(m, SMaxpool2D) or isinstance(m, SReLU):
                new = FakeSModule(m.op)
                self._modules[name] = new
        self.to(device)
    
    def to_first_only(self):
        self.first_only = True
        for m in self.modules():
            if isinstance(m, SModel):
                m.first_only = True
        for n, m in self.named_modules():
            if isinstance(m, SModule):
                n = n.split(".")
                father = self
                for i in range(len(n) - 1):
                    father = father._modules[n[i]]
                mo = father._modules[n[-1]]
                new = mo.copy_N()
                father._modules[n[-1]] = new
            if isinstance(m, SReLU) or isinstance(m, SMaxpool2D) or isinstance(m, SBatchNorm2d) or isinstance(m, SAdaptiveAvgPool2d):
                n = n.split(".")
                father = self
                for i in range(len(n) - 1):
                    father = father._modules[n[i]]
                father._modules[n[-1]] = m.op
                if isinstance(m, SMaxpool2D):
                    father._modules[n[-1]].return_indices = False

    def from_first_back_second(self):
        self.first_only = False
        for m in self.modules():
            if isinstance(m, SModel):
                m.first_only = False
        for n, m in self.named_modules():
            if isinstance(m, NModule):
                n = n.split(".")
                father = self
                for i in range(len(n) - 1):
                    father = father._modules[n[i]]
                mo = father._modules[n[-1]]
                new = mo.copy_S()
                father._modules[n[-1]] = new
            if isinstance(m, nn.ReLU) or isinstance(m, nn.MaxPool2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.AdaptiveAvgPool2d):
                n = n.split(".")
                father = self
                for i in range(len(n) - 1):
                    father = father._modules[n[i]]
                if isinstance(m, nn.ReLU):
                    new = SReLU(m.inplace)
                elif isinstance(m, nn.MaxPool2d):
                    new = SMaxpool2D(m.kernel_size, m.stride, m.padding, m.dilation, True, m.ceil_mode)
                elif isinstance(m, nn.AdaptiveAvgPool2d):
                    new = SAdaptiveAvgPool2d(m.output_size)
                    new.op = m
                elif isinstance(m, nn.BatchNorm2d):
                    new = SBatchNorm2d(m.num_features)
                    new.op = m
                # TODO: Other modules specified above
                father._modules[n[-1]] = new

    def normalize(self):
        for mo in self.modules():
            if isinstance(mo, SLinear) or isinstance(mo, SConv2d):
                mo.normalize()

    def get_scale(self):
        scale = 1.0
        for m in self.modules():
            if isinstance(m, SModule):
                scale *= m.scale
        return scale

    def de_normalize(self):
        for mo in self.modules():
            if isinstance(mo, SLinear) or isinstance(mo, SConv2d):
                if mo.original_w is None:
                    raise Exception("no original weight")
                else:
                    mo.scale = 1.0
                    mo.op.weight.data = mo.original_w
                    mo.original_w = None
                    if mo.original_b is not None:
                        mo.op.bias.data = mo.original_b
                        mo.original_b = None
    
    def back_real(self, device):
        for name, m in self.named_modules():
            if isinstance(m, FakeSModule):
                if isinstance(m.op, nn.Linear):
                    if m.op.bias is not None:
                        bias = True
                    new = SLinear(m.op.in_features, m.op.out_features, bias)
                    new.op = m.op
                    self._modules[name] = new

                elif isinstance(m.op, nn.Conv2d):
                    if m.op.bias is not None:
                        bias = True
                    new = SConv2d(m.op.in_channels, m.op.out_channels, m.op.kernel_size, m.op.stride, m.op.padding, m.op.dilation, m.op.groups, bias, m.op.padding_mode)
                    new.op = m.op
                    self._modules[name] = new

                elif isinstance(m.op, nn.MaxPool2d):
                    new = SMaxpool2D(m.op.kernel_size, m.op.stride, m.op.padding, m.op.dilation, m.op.return_indices, m.op.ceil_mode)
                    new.op = m.op
                    new.op.return_indices = True
                    self._modules[name] = new

                elif isinstance(m.op, nn.ReLU):
                    new = SReLU()
                    new.op = m.op
                    self._modules[name] = new

                else:
                    raise NotImplementedError
        self.to(device)
