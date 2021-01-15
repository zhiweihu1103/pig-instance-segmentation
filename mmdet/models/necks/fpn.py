import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.core import auto_fp16
from ..registry import NECKS
from ..utils import ConvModule
from ..attentions import CS_CBAM_Module
from ..attentions import BAM
from ..attentions import csSE
from ..attentions import S_Module
from ..attentions import C_Module
from ..attentions import CS_Module

@NECKS.register_module
class FPN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

		# attention blocks list
        self.cs_cbams = nn.ModuleList()
        self.cs_bams = nn.ModuleList()
        self.cs_scses = nn.ModuleList()
        self.s_attentions = nn.ModuleList()
        self.c_attentions = nn.ModuleList()
        self.cs_attentions = nn.ModuleList()
        self.acnet_blocks = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                activation=self.activation,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)
			# cbam attention blocks
            cs_cbam = CS_CBAM_Module(out_channels)

            # bam attention blocks
            cs_bam = BAM(out_channels)

            # scse attention blocks
            cs_scse = csSE(out_channels)

            # dab attention blocks
            acnet_block = ACBlock(out_channels, out_channels, padding=1)
            s_attention = S_Module(out_channels)
            c_attention = C_Module(out_channels)
            # cs_ccattention = CS_CCModule(out_channels)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

            self.cs_cbams.append(cs_cbam)
            self.cs_bams.append(cs_bam)
            self.cs_scses.append(cs_scse)

            self.acnet_blocks.append(acnet_block)
            self.s_attentions.append(s_attention)
            self.c_attentions.append(c_attention)
            # self.cs_ccattentions.append(cs_ccattention)
            
        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    activation=self.activation,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]   

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # cbam attention blocks
            # cs_cbam_feature = self.cs_cbams[i](laterals[i - 1], F.interpolate(laterals[i], scale_factor=2, mode='nearest'))
            # laterals[i - 1] = laterals[i - 1] + cs_cbam_feature
			
			# bam attention blocks
            # cs_bam_feature = self.cs_bams[i](laterals[i - 1], F.interpolate(laterals[i], scale_factor=2, mode='nearest'))
            # laterals[i - 1] = laterals[i - 1] + cs_bam_feature
            
            # scse attention blocks
            # cs_scse_feature = self.cs_scses[i](laterals[i - 1], F.interpolate(laterals[i], scale_factor=2, mode='nearest'))
            # laterals[i - 1] = laterals[i - 1] + cs_scse_feature

            # dab attention blocks (sab+cab)
            merge = laterals[i - 1] + F.interpolate(laterals[i], scale_factor=2, mode='nearest')
            merge = self.acnet_blocks[i](merge)
            # change the recurrence value to modify the attention layers
            recurrence = 2
            for j in range(recurrence):
                merge_s = self.s_attentions[i](merge)
            merge_c = self.c_attentions[i](merge)
            laterals[i - 1] = merge_s + merge_c

            # dab attention blocks (sab)
            # merge = laterals[i - 1] + F.interpolate(laterals[i], scale_factor=2, mode='nearest')
            # merge = self.acnet_blocks[i](merge)
            # recurrence = 2
            # for j in range(recurrence):
            #     merge = self.s_attentions[i](merge)
            # laterals[i - 1] = merge

            # dab attention blocks (cab)
            # merge = laterals[i - 1] + F.interpolate(laterals[i], scale_factor=2, mode='nearest')
            # merge = self.acnet_blocks[i](merge)
            # laterals[i - 1] = self.c_attentions[i](merge)

            # normal (not attention blocks)
            # laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], scale_factor=2, mode='nearest')

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                else:
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)

# ACNet part
class CropLayer(nn.Module):
    def __init__(self, crop_set):
        super(CropLayer, self).__init__()
        self.rows_to_crop = - crop_set[0]
        self.cols_to_crop = - crop_set[1]
        assert self.rows_to_crop >= 0
        assert self.cols_to_crop >= 0

    def forward(self, input):
        return input[:, :, self.rows_to_crop:-self.rows_to_crop, self.cols_to_crop:-self.cols_to_crop]

class ACBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False):
        super(ACBlock, self).__init__()
        self.deploy = deploy
        if deploy:
            self.fused_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size,kernel_size), stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
        else:
            self.square_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(kernel_size, kernel_size), stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=False,
                                         padding_mode=padding_mode)
            self.square_bn = nn.BatchNorm2d(num_features=out_channels)

            center_offset_from_origin_border = padding - kernel_size // 2
            ver_pad_or_crop = (center_offset_from_origin_border + 1, center_offset_from_origin_border)
            hor_pad_or_crop = (center_offset_from_origin_border, center_offset_from_origin_border + 1)
            if center_offset_from_origin_border >= 0:
                self.ver_conv_crop_layer = nn.Identity()
                ver_conv_padding = ver_pad_or_crop
                self.hor_conv_crop_layer = nn.Identity()
                hor_conv_padding = hor_pad_or_crop
            else:
                self.ver_conv_crop_layer = CropLayer(crop_set=ver_pad_or_crop)
                ver_conv_padding = (0, 0)
                self.hor_conv_crop_layer = CropLayer(crop_set=hor_pad_or_crop)
                hor_conv_padding = (0, 0)
            self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 1),
                                      stride=stride,
                                      padding=ver_conv_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)

            self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3),
                                      stride=stride,
                                      padding=hor_conv_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)
            self.ver_bn = nn.BatchNorm2d(num_features=out_channels)
            self.hor_bn = nn.BatchNorm2d(num_features=out_channels)
            self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        if self.deploy:
            return self.fused_conv(input)
        else:
            square_outputs = self.square_conv(input)
            square_outputs = self.square_bn(square_outputs)
            vertical_outputs = self.ver_conv_crop_layer(input)
            vertical_outputs = self.ver_conv(vertical_outputs)
            vertical_outputs = self.ver_bn(vertical_outputs)
            horizontal_outputs = self.hor_conv_crop_layer(input)
            horizontal_outputs = self.hor_conv(horizontal_outputs)
            horizontal_outputs = self.hor_bn(horizontal_outputs)
            return self.relu(square_outputs + vertical_outputs + horizontal_outputs)