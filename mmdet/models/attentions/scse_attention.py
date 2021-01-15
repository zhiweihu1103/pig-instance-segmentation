"""
@author: zhiweihu
@create time: 2020-3-28 23:37
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class sSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        q = self.Conv1x1(U)  # U:[bs,c,h,w] to q:[bs,1,h,w]
        q = self.norm(q)
        # concate_z = concate.permute(0, 3, 1, 2)
        self.s_csse_attention_map = F.interpolate(q, size=[1024, 2048],mode='bilinear', align_corners=True)
        return U * q  # 广播机制

class cSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.Conv_Squeeze = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False)
        self.Conv_Excitation = nn.Conv2d(in_channels//2, in_channels, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        z = self.avgpool(U)# shape: [bs, c, h, w] to [bs, c, 1, 1]
        z = self.Conv_Squeeze(z) # shape: [bs, c/2]
        z = self.Conv_Excitation(z) # shape: [bs, c]
        z = self.norm(z)
        return U * z.expand_as(U)

class csSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.first_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)
        self.last_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)
        self.cSE = cSE(in_channels)
        self.sSE = sSE(in_channels)

    def forward(self, lower, higher):
        input = lower + higher
        U = self.first_conv(input)
        U_sse = self.sSE(U)
        U_cse = self.cSE(U)
        return U_cse+U_sse