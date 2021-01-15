"""
@author: zhiweihu
@create time: 2020-3-27 16:13
"""

import numpy as np
import torch
import math
import torch.nn as nn
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable
torch_ver = torch.__version__[:3]

__all__ = ['S_CBAM_Module', 'C_CBAM_Module', 'CS_CBAM_Module']

class S_CBAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim, kernel_size=7):
        super(S_CBAM_Module, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 4 if kernel_size == 7 else 1
        
        self.conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3)
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self,lower, higher):
        merge = lower + higher
        x = self.conv(merge)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.sigmoid(x)
        self.cbam_s = x
        out = merge * x + merge
        return out


class C_CBAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim, reduction=8):
        super(C_CBAM_Module, self).__init__()
        self.conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_dim, in_dim // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_dim // reduction, in_dim, kernel_size=1, padding=0)
        self.sigmoid_channel = nn.Sigmoid()
    def forward(self,lower, higher):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        merge = lower + higher
        y = self.conv(merge)
        avg = self.avg_pool(y)
        mx = self.max_pool(y)
        avg = self.fc1(avg)
        mx = self.fc1(mx)
        avg = self.relu(avg)
        mx = self.relu(mx)
        avg = self.fc2(avg)
        mx = self.fc2(mx)
        x = avg + mx
        x = self.sigmoid_channel(x)
        out = merge * x + merge
        return out

class CS_CBAM_Module(Module):
    def __init__(self, in_dim, reduction=8, kernel_size=7):
        super(CS_CBAM_Module, self).__init__()
        self.conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_dim, in_dim // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_dim // reduction, in_dim, kernel_size=1, padding=0)
        self.sigmoid_channel = nn.Sigmoid()
        
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 4 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        self.last_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3, padding=1)
        
    def forward(self,lower, higher):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        merge = lower + higher
        y = self.conv(merge)
        avg = self.avg_pool(y)
        mx = self.max_pool(y)
        avg = self.fc1(avg)
        mx = self.fc1(mx)
        avg = self.relu(avg)
        mx = self.relu(mx)
        avg = self.fc2(avg)
        mx = self.fc2(mx)
        x_cam = avg + mx
        x_cam = self.sigmoid_channel(x_cam)
        pam_out = merge * x_cam
        
        avg_out = torch.mean(y, dim=1, keepdim=True)
        max_out, _ = torch.max(y, dim=1, keepdim=True)
        x_sam = torch.cat([avg_out, max_out], dim=1)
        x_sam = self.conv1(x_sam)
        x_sam = self.sigmoid(x_sam)
        pam_sam_out = pam_out * x_sam
        out = pam_sam_out + merge
        out = self.last_conv(out)
        return out

