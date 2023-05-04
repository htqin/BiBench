import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.parameter import Parameter


class FDA_BinaryQuantize(Function):
    @staticmethod
    def forward(ctx, inputs, n):
        ctx.save_for_backward(inputs, n)
        out = torch.sign(inputs)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        inputs, n = ctx.saved_tensors
        omega = 0.1
        grad_input = 4 * omega / np.pi * sum(([torch.cos((2 * i + 1) * omega * inputs) for i in range(n + 1)])) * grad_output
        grad_input[inputs.gt(1)] = 0
        grad_input[inputs.lt(-1)] = 0
        return grad_input, None


class NoiseAdaption(nn.Module):

    def __init__(self, d, k=64):
        super(NoiseAdaption, self).__init__()
        self.fc1 = nn.Linear(in_features=d, out_features=d // k, bias=False)
        self.fc2 = nn.Linear(in_features=d // k, out_features=d, bias=False)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        shape = x.shape
        _x = x
        x = x.view(shape[0], -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).reshape(shape)
        x += 0.1 * torch.sin(_x)
        return x


class NoiseAdaption_a(nn.Module):

    def __init__(self, d, k=64):
        super(NoiseAdaption_a, self).__init__()
        self.fc1 = nn.Linear(in_features=d, out_features=d // k, bias=False)
        self.fc2 = nn.Linear(in_features=d // k, out_features=d, bias=False)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        shape = x.shape
        _x = x
        x = x.view(shape[0], shape[1], -1).contiguous()
        x = x.permute(0, 2, 1).contiguous()
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = x.permute(0, 2, 1).contiguous().reshape(shape)
        x += 0.1 * torch.sin(_x)
        return x


class FDAConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, noise_adapt=True):
        super(FDAConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.register_buffer('n', torch.tensor(9))
        self.register_buffer('alpha', torch.tensor(0.1))
        self.noise_adapt = noise_adapt
        d, k = int(torch.prod(torch.tensor(self.weight.shape[1:])).numpy()), 64
        if self.noise_adapt:
            self.act_noise = None
            try:
                self.weight_noise = NoiseAdaption(d=d, k=64)
            except:
                self.weight_noise = NoiseAdaption(d=d, k=4)

    def model_params_update(self, max_epochs, epoch):
        alpha = 0.05 * (1 + torch.cos(torch.tensor((epoch + 1) / max_epochs) * np.pi))
        self.alpha = alpha.to(self.alpha.device)
        self.n = torch.tensor(9 + int(epoch / max_epochs * 9)).to(self.n.device)

    def forward(self, input):
        w = self.weight
        a = input
        bw = FDA_BinaryQuantize().apply(w, self.n)
        ba = FDA_BinaryQuantize().apply(a, self.n)

        if self.noise_adapt:
            if self.act_noise is None:
                d = int(torch.prod(torch.tensor(input.shape[1])).numpy())
                try:
                    self.act_noise = NoiseAdaption_a(d=d, k=64).to(input.device)
                except:
                    self.act_noise = NoiseAdaption_a(d=d, k=4).to(input.device)

            ew = self.weight_noise(w) 
            bw = bw + self.alpha * ew

            ea = self.act_noise(a)
            ba = ba + self.alpha * ea
       
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(self.weight),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        scaling_factor = scaling_factor.detach()
        bw = bw * scaling_factor
        output = F.conv2d(ba, bw, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        return output


class FDAConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, noise_adapt=True):
        super(FDAConv1d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.register_buffer('n', torch.tensor(9))
        self.register_buffer('alpha', torch.tensor(0.1))
        self.noise_adapt = noise_adapt
        d, k = int(torch.prod(torch.tensor(self.weight.shape[1:])).numpy()), 64
        if self.noise_adapt:
            self.act_noise = None
            try:
                self.weight_noise = NoiseAdaption(d=d, k=64)
            except:
                self.weight_noise = NoiseAdaption(d=d, k=4)

    def model_params_update(self, max_epochs, epoch):
        alpha = 0.05 * (1 + torch.cos(torch.tensor((epoch + 1) / max_epochs) * np.pi))
        self.alpha = alpha.to(self.alpha.device)
        self.n = torch.tensor(9 + int(epoch / max_epochs * 9)).to(self.n.device)

    def forward(self, input):
        w = self.weight
        a = input
        bw = FDA_BinaryQuantize().apply(w, self.n)
        ba = FDA_BinaryQuantize().apply(a, self.n)

        if self.noise_adapt:
            if self.act_noise is None:
                d = int(torch.prod(torch.tensor(input.shape[1])).numpy())
                try:
                    self.act_noise = NoiseAdaption_a(d=d, k=64).to(input.device)
                except:
                    self.act_noise = NoiseAdaption_a(d=d, k=4).to(input.device)

            ew = self.weight_noise(w) 
            bw = bw + self.alpha * ew

            ea = self.act_noise(a)
            ba = ba + self.alpha * ea
       
        scaling_factor = torch.mean(torch.mean(abs(self.weight),dim=2,keepdim=True),dim=1,keepdim=True)
        scaling_factor = scaling_factor.detach()
        bw = bw * scaling_factor
        output = F.conv1d(ba, bw, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        return output

class FDALinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, binary_act=True):
        super(FDALinear, self).__init__(in_features, out_features, bias=bias)
        self.register_buffer('n', torch.tensor(9))
        self.register_buffer('alpha', torch.tensor(0.1))
        self.noise_adapt = True
        d, k = int(torch.prod(torch.tensor(self.weight.shape[1:])).numpy()), 64
        if self.noise_adapt:
            self.act_noise = None
            self.weight_noise = NoiseAdaption(d=d, k=64)
 
    def model_params_update(self, max_epochs, epoch):
        alpha = 0.05 * (1 + torch.cos(torch.tensor((epoch + 1) / max_epochs) * np.pi))
        self.alpha = alpha.to(self.alpha.device)
        self.n = torch.tensor(9 + int(epoch / max_epochs * 9)).to(self.n.device)

    def forward(self, input, type=None):
        
        shape = input.shape
        a = input.view(-1, shape[-1])
        w = self.weight
        bw = FDA_BinaryQuantize().apply(w, self.n)
        ba = FDA_BinaryQuantize().apply(a, self.n)

        if self.noise_adapt:
            if self.act_noise is None:
                d = int(torch.prod(torch.tensor(a.shape[1])).numpy())
                self.act_noise = NoiseAdaption(d=d, k=64).to(input.device)

            ew = self.weight_noise(w) 
            bw = bw + self.alpha * ew

            ea = self.act_noise(a)
            ba = ba + self.alpha * ea
       
        scaling_factor = torch.abs(self.weight).mean(dim=1, keepdim=True)
        scaling_factor = scaling_factor.detach()
        bw = bw * scaling_factor
        out = nn.functional.linear(ba, bw)
        out = out.view(*shape[:-1], -1)
        
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out) 

        return out
