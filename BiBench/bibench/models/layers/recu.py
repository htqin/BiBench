import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.autograd import Function, Variable


class ReCUConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(ReCUConv2d, self).__init__(*kargs, **kwargs)
        self.alpha = nn.Parameter(torch.rand(self.weight.size(0), 1, 1), requires_grad=True)
        self.register_buffer('tau', torch.tensor(1.))

    def model_params_update(self, max_epochs, epoch):
        "compute tau"
        a = torch.tensor(np.e)
        T_min, T_max = torch.tensor(0.85).float(), torch.tensor(0.99).float()
        A = (T_max - T_min) / (a - 1)
        B = T_min - A
        tau = A * torch.tensor([torch.pow(a, epoch / max_epochs)]).float() + B
        self.tau = tau.to(self.tau.device)

    def forward(self, input):
        a = input
        w = self.weight

        w0 = w - w.mean([1,2,3], keepdim=True)
        w1 = w0 / (torch.sqrt(w0.var([1,2,3], keepdim=True) + 1e-5) / 2 / np.sqrt(2))
        EW = torch.mean(torch.abs(w1))
        Q_tau = (- EW * torch.log(2-2*self.tau)).detach().cpu().item()
        w2 = torch.clamp(w1, -Q_tau, Q_tau)

        if self.training:
            a0 = a / torch.sqrt(a.var([1,2,3], keepdim=True) + 1e-5)
        else: 
            a0 = a
        
        #* binarize
        bw = BinaryQuantize().apply(w2)
        ba = BinaryQuantize_a().apply(a0)
        #* 1bit conv
        output = F.conv2d(ba, bw, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        #* scaling factor
        output = output * self.alpha
        return output


class ReCUConv1d(nn.Conv1d):

    def __init__(self, *kargs, **kwargs):
        super(ReCUConv1d, self).__init__(*kargs, **kwargs)
        self.alpha = nn.Parameter(torch.rand(self.weight.size(0), 1), requires_grad=True)
        self.register_buffer('tau', torch.tensor(1.))

    def model_params_update(self, max_epochs, epoch):
        "compute tau"
        a = torch.tensor(np.e)
        T_min, T_max = torch.tensor(0.85).float(), torch.tensor(0.99).float()
        A = (T_max - T_min) / (a - 1)
        B = T_min - A
        tau = A * torch.tensor([torch.pow(a, epoch / max_epochs)]).float() + B
        self.tau = tau.to(self.tau.device)

    def forward(self, input):
        a = input
        w = self.weight

        w0 = w - w.mean([1,2], keepdim=True)
        w1 = w0 / (torch.sqrt(w0.var([1,2], keepdim=True) + 1e-5) / 2 / np.sqrt(2))
        EW = torch.mean(torch.abs(w1))
        Q_tau = (- EW * torch.log(2-2*self.tau)).detach().cpu().item()
        w2 = torch.clamp(w1, -Q_tau, Q_tau)

        if self.training:
            a0 = a / torch.sqrt(a.var([1,2], keepdim=True) + 1e-5)
        else: 
            a0 = a
        
        #* binarize
        bw = BinaryQuantize().apply(w2)
        ba = BinaryQuantize_a().apply(a0)
        #* 1bit conv
        output = F.conv1d(ba, bw, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        #* scaling factor
        output = output * self.alpha
        return output


class BinaryQuantize(Function):
    @staticmethod
    def forward(ctx, input):
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class BinaryQuantize_a(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_input = (2 - torch.abs(2*input))
        grad_input = grad_input.clamp(min=0) * grad_output.clone()
        return grad_input


class BinaryQuantizeW(Function):
    @staticmethod
    def forward(ctx, input):
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class BinaryQuantizeA(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_input = (2 - torch.abs(2*input))
        grad_input = grad_input.clamp(min=0) * grad_output.clone()
        return grad_input


class ReCULinear(nn.Linear):
    def __init__(self,  in_features, out_features, bias=True, binary_act=True):
        super(ReCULinear, self).__init__(in_features, out_features, bias=bias)
        self.alpha = nn.Parameter(torch.rand(1, 1, self.weight.size(0)), requires_grad=True)
        self.register_buffer('tau', torch.tensor(1.))
 
    def model_params_update(self, max_epochs, epoch):
        "compute tau"
        a = torch.tensor(np.e)
        T_min, T_max = torch.tensor(0.85).float(), torch.tensor(0.99).float()
        A = (T_max - T_min) / (a - 1)
        B = T_min - A
        tau = A * torch.tensor([torch.pow(a, epoch / max_epochs)]).float() + B
        self.tau = tau.to(self.tau.device)

    def forward(self, input, type=None):
        shape = input.shape
        a = input.view(-1, input.shape[-1])

        w = self.weight
        w0 = w - w.mean([1], keepdim=True)
        w1 = w0 / (torch.sqrt(w0.var([1], keepdim=True) + 1e-5) / 2 / np.sqrt(2))
        EW = torch.mean(torch.abs(w1))
        Q_tau = (- EW * torch.log(2-2*self.tau)).detach().cpu().item()
        w2 = torch.clamp(w1, -Q_tau, Q_tau)

        if self.training:
            a0 = a / torch.sqrt(a.var([1], keepdim=True) + 1e-5)
        else: 
            a0 = a

        bw = BinaryQuantizeW().apply(w2)
        ba = BinaryQuantizeA().apply(a0)
        #* 1bit conv
        out = F.linear(ba, bw)
        out = out * self.alpha

        out = out.view(*shape[:-1], -1)

        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out) 

        return out
