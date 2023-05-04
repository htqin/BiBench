import torch
import torch.nn as nn
import torch.nn.functional as F


class XNORConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False, dilation=0, transposed=False, output_padding=None, groups=1):
        super(XNORConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.number_of_weights = in_channels * out_channels * kernel_size * kernel_size
        self.shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.rand(*self.shape) * 0.001, requires_grad=True)

    def forward(self, x):
        real_input = x
        scaling_factor_x = torch.mean(torch.mean(torch.mean(abs(real_input),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        scaling_factor_x = scaling_factor_x.detach()
        binary_input_no_grad = scaling_factor_x * torch.sign(x)
        cliped_input = torch.clamp(x, -1.0, 1.0)
        x = binary_input_no_grad.detach() - cliped_input.detach() + cliped_input

        real_weights = self.weight.view(self.shape)
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)

        return y


class XNORConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False, dilation=0, transposed=False, output_padding=None, groups=1):
        super(XNORConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.number_of_weights = in_channels * out_channels * kernel_size
        self.shape = (out_channels, in_channels, kernel_size)
        self.weight = nn.Parameter(torch.rand(*self.shape) * 0.001, requires_grad=True)

    def forward(self, x):
        real_input = x
        scaling_factor_x = torch.mean(torch.mean(abs(real_input),dim=2,keepdim=True),dim=1,keepdim=True)
        scaling_factor_x = scaling_factor_x.detach()
        binary_input_no_grad = scaling_factor_x * torch.sign(x)
        cliped_input = torch.clamp(x, -1.0, 1.0)
        x = binary_input_no_grad.detach() - cliped_input.detach() + cliped_input

        real_weights = self.weight.view(self.shape)
        scaling_factor = torch.mean(torch.mean(abs(real_weights),dim=2,keepdim=True),dim=1,keepdim=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        y = F.conv1d(x, binary_weights, stride=self.stride, padding=self.padding)

        return y


class BinaryQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output
        grad_input[input[0].gt(1)] = 0
        grad_input[input[0].lt(-1)] = 0
        return grad_input


class XNORLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, binary_act=True):
        super(XNORLinear, self).__init__(in_features, out_features, bias=bias)
        self.binary_act = binary_act

    def forward(self, input):
        bw = self.weight
        ba = input
        bw = bw - bw.mean(-1).view(-1, 1)
        sw = bw.abs().mean(-1).view(-1, 1).detach()
        bw = BinaryQuantize().apply(bw)
        bw = bw * sw
        if self.binary_act:
            sa = ba.abs().mean(-1).view(-1, 1).detach()
            ba = BinaryQuantize().apply(ba)
            ba = ba * sa
        output = F.linear(ba, bw, self.bias)
        return output
