import math
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

class depthwise_conv(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1):
        super(depthwise_conv, self).__init__()
        self.depthwise = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        # support for 4D tensor with NCHW
        C, H, W = x.shape[1:]
        x = x.reshape(-1, 1, H, W)
        x = self.depthwise(x)
        x = x.view(-1, C, H, W)
        return x


class depthwise_block(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, activation='relu'):
        super(depthwise_block, self).__init__()
        self.depthwise = depthwise_conv(kernel_size=3, stride=1, padding=1)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()

    def forward(self, x, act=True):
        x = self.depthwise(x)
        if act:
            x = self.activation(x)
        return x


class bottleneck_block(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, activation='relu'):
        super(bottleneck_block, self).__init__()
        self.depthwise = depthwise_conv(kernel_size=3, stride=1, padding=1)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()


    def forward(self, x, act=True):
        sum_layer = x.max(dim=1, keepdim=True)[0] #因此，sum_layer 表示 x 在第1个维度上的最大值，并保持了维度的一致性。这通常用于一些池化操作或特征提取中，以便获取每个样本的最大值特征。
        x = self.depthwise(x)
        x = x + sum_layer
        if act:
            x = self.activation(x)
        return x
    

if __name__ == "__main__":
    depthconve = depthwise_conv(kernel_size=3, stride=1, padding=1)

    depthwise_block_test = depthwise_block()

    bottleneck_block_test = bottleneck_block()

    input_tensor = torch.ones((2, 32, 256, 256))

    depth_block_output = depthwise_block_test(input_tensor)
    print(depth_block_output.shape) #torch.Size([2, 32, 256, 256])

    bottleneck_block_output = bottleneck_block_test(input_tensor)
    print(bottleneck_block_output.shape) #torch.Size([2, 32, 256, 256])