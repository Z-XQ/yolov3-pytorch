# -*- coding: utf-8 -*-
# @Time    : 2020/10/22 下午10:10
# @Author  : zxq
# @File    : yolov3.py
# @Software: PyCharm

import torch.nn as nn

from backbone.darknet53 import Conv2dBatchLeaky, darknet53


class Conv2DBlock5L(nn.Module):
    """
    对应网络结构图中的Conv2D Block 5L，具体功能是6个conv+bn+leakyReLU，
    为什么叫5L，我猜是输出通道有5次是在c1和c2之间变换
    只改变通道数
    """
    def __init__(self, in_channels, out_channels):
        """
        :param in_channels: 前面DarkNet输出的特征图通道数
        :param out_channels: list. [c1, c2]. 通道数就在c1和c2之间变化，最后输出
        然后5个卷积的通道数就在in_channels和in_channels//2两者间变化
        """
        self.conv = Conv2dBatchLeaky(in_channels=in_channels, out_channels=out_channels[0], kernel_size=1, stride=1)  # 降维，减少计算量

        self.conv1 = Conv2dBatchLeaky(in_channels=out_channels[0], out_channels=out_channels[1], kernel_size=3, stride=1)
        self.conv2 = Conv2dBatchLeaky(in_channels=out_channels[1], out_channels=out_channels[0], kernel_size=1, stride=1)
        self.conv3 = Conv2dBatchLeaky(in_channels=out_channels[0], out_channels=out_channels[1], kernel_size=3, stride=1)
        self.conv4 = Conv2dBatchLeaky(in_channels=out_channels[1], out_channels=out_channels[0], kernel_size=1, stride=1)
        self.conv5 = Conv2dBatchLeaky(in_channels=out_channels[0], out_channels=out_channels[1], kernel_size=3, stride=1)

        # 打包下，省得在forward重复写
        self.layers = nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class YOLOv3(nn.Module):
    def __init__(self):
        super(YOLOv3, self).__init__()
        self.backbone = darknet53(pretrained=False)

        Conv2DBlock5L(in_channels=)