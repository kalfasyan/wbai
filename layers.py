from models import ModuleWrapper
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from metrics import calculate_kl

class BBBConv1d(ModuleWrapper):
    
    def __init__(self, in_channels, out_channels, kernel_size, alpha_shape, stride=1,
                 padding=0, dilation=1, bias=True, name='BBBConv1d'):
        super(BBBConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size          # 2D: (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.alpha_shape = alpha_shape
        self.groups = 1
        self.weight = Parameter(torch.Tensor(
            out_channels, in_channels, self.kernel_size))               # 2D: *self.kernel_size
        if bias:
            self.bias = Parameter(torch.Tensor(1, out_channels, 1))  # 2D: (1, out_channels, 1, 1)
        else:
            self.register_parameter('bias', None)
        self.out_bias = lambda input, kernel: F.conv1d(input, kernel, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self.out_nobias = lambda input, kernel: F.conv1d(input, kernel, None, self.stride, self.padding, self.dilation, self.groups)
        self.log_alpha = Parameter(torch.Tensor(*alpha_shape))
        self.reset_parameters()
        self.name = name


    def reset_parameters(self):
        n = self.in_channels
        # for k in self.kernel_size:
        #     n *= k
        n *= self.kernel_size
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        self.log_alpha.data.fill_(-5.0)

    def forward(self, x):

        mean = self.out_bias(x, self.weight)

        sigma = torch.exp(self.log_alpha) * self.weight * self.weight

        std = torch.sqrt(1e-16 + self.out_nobias(x * x, sigma))
        # if self.training:
        epsilon = std.data.new(std.size()).normal_()
        # else:
        #     epsilon = 0.0

        # Local reparameterization trick
        out = mean + std * epsilon

        return out

    def kl_loss(self):
        return self.weight.nelement() / self.log_alpha.nelement() * calculate_kl(self.log_alpha)

class BBBLinear(ModuleWrapper):
    
    def __init__(self, in_features, out_features, alpha_shape=(1, 1), bias=True, name='BBBLinear'):
        super(BBBLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha_shape = alpha_shape
        self.W = Parameter(torch.Tensor(out_features, in_features))
        self.log_alpha = Parameter(torch.Tensor(*alpha_shape))
        if bias:
            self.bias = Parameter(torch.Tensor(1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.kl_value = calculate_kl
        self.name = name


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        self.log_alpha.data.fill_(-5.0)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x):

        mean = F.linear(x, self.W)
        if self.bias is not None:
            mean = mean + self.bias

        sigma = torch.exp(self.log_alpha) * self.W * self.W

        std = torch.sqrt(1e-16 + F.linear(x * x, sigma))
        # if self.training:
        epsilon = std.data.new(std.size()).normal_()
        # else:
        #     epsilon = 0.0
        # Local reparameterization trick
        out = mean + std * epsilon

        return out

    def kl_loss(self):
        return self.W.nelement() * self.kl_value(self.log_alpha) / self.log_alpha.nelement()