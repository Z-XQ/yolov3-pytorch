# -*- coding: utf-8 -*-
# @Time    : 2020/10/20 下午10:17
# @Author  : zxq
# @File    : YOLOv3_model.py
# @Software: PyCharm
from collections import OrderedDict

import torch.nn as nn


class Conv2dBatchLeaky(nn.Module):
    """ This convenience layer groups a 2D convolution, a batchnorm and a leaky ReLU.
    They are executed in a sequential manner.
    DarkNet最小子模块
    只有stride=1控制特征缩放

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int or tuple): Size of the kernel of the convolution
        stride (int or tuple): Stride of the convolution
        negative_slope (number, optional): Controls the angle of the negative slope of the leaky ReLU; Default **0.1**
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, negative_slope=0.1):
        super(Conv2dBatchLeaky, self).__init__()

        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        # padding, 所以如果stride=1,则不会改变特征的高宽
        if isinstance(kernel_size, (list, tuple)):
            self.padding = [int(ii / 2) for ii in kernel_size]
        else:
            self.padding = int(kernel_size / 2)  # 向下取整
        self.leaky_slope = negative_slope

        # Layer，打包
        self.layers = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=False),
            nn.BatchNorm2d(self.out_channels),  # , eps=1e-6, momentum=0.01),
            nn.LeakyReLU(self.leaky_slope, inplace=True)
        )

    def __repr__(self):
        """
        打印class时，其实打印的就是这个function的返回值
        :return:
        """
        s = '{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}, ' \
            'negative_slope={leaky_slope}) '
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, x):
        x = self.layers(x)  # 因为打包好了，这里只需一句搞定
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels):
        """
        残差块
        每个BasizeBlock由两次conv+bn+leakyReLU组成
        特征图的通道数变化： in_channels -> in_channels//2 -> in_channels
        :param in_channels: 输入x特征图的通道数
        """
        super(ResBlock, self).__init__()
        # in_channels -> in_channels // 2，channel维度降维，减少参数目的，这也是为什么两次卷积后再残差的原因。
        self.conv1 = Conv2dBatchLeaky(in_channels, in_channels // 2, kernel_size=1, stride=1, negative_slope=0.1)
        # in_channels//2 -> in_channels
        self.conv2 = Conv2dBatchLeaky(in_channels // 2, in_channels, kernel_size=3, stride=1, negative_slope=0.1)

    def forward(self, x):
        input_feature = x  # in_channels = 64, 则out_channels=32
        x = self.conv1(x)  # -> in_channels//2
        x = self.conv2(x)  # -> channels=64
        x += input_feature  # 残差块：输入的特征加上两次卷积后的特征，作为下一个残差块的输入。
        return x


class DarkNet(nn.Module):
    def __init__(self, layers):
        """
        DarkNet由5个模块组成，每个模块又由多个残差块组成
        :param layers: list. len(layers)==5，每个数字代表各个模块的残差块个数，可以用来控制模型的大小。
        eg.
        Darknet53, layers==[1, 2, 8, 8, 4]
        """
        super(DarkNet, self).__init__()
        start_channel = 32  # 第一个卷积后的特征图通道数，这里固定  # c= 32
        self.conv = Conv2dBatchLeaky(in_channels=3, out_channels=start_channel, kernel_size=3, stride=1)  # 高宽不变

        # 定义5个模块，每个模块前面都有一个卷积用于高宽的下采样，同时通道数翻倍。每个模块不会改变特征维度，包括h,w,c。
        self.conv1 = Conv2dBatchLeaky(in_channels=start_channel, out_channels=start_channel * 2, kernel_size=3,
                                      stride=2)  # 32->64
        self.layer1 = self._build_layer(input_channels=self.start_channel * 2, num_res_block=layers[0])  # 64->64

        self.conv2 = Conv2dBatchLeaky(in_channels=start_channel * 2, out_channels=start_channel * 4, kernel_size=3,
                                      stride=2)  # ->128
        self.layer2 = self._build_layer(input_channels=start_channel * 4, num_res_block=layers[1])  # 128->128

        self.conv3 = Conv2dBatchLeaky(in_channels=start_channel * 4, out_channels=start_channel * 8, kernel_size=3,
                                      stride=2)  # ->256
        self.layer3 = self._build_layer(input_channels=start_channel * 8, num_res_block=layers[2])  # 256->256

        self.conv4 = Conv2dBatchLeaky(in_channels=start_channel * 8, out_channels=start_channel * 16, kernel_size=3,
                                      stride=2)  # ->512
        self.layer4 = self._build_layer(input_channels=start_channel * 16, num_res_block=layers[3])  # 512->512

        self.conv4 = Conv2dBatchLeaky(in_channels=start_channel * 16, out_channels=start_channel * 32, kernel_size=3,
                                      stride=2)  # ->1024
        self.layer5 = self._build_layer(input_channels=start_channel * 32, num_res_block=layers[4])  # 1024->1024

    @staticmethod
    def _build_layer(input_channels, num_res_block=1):
        """
        建议DarkNet子模块
        每个子模块都是由多个残差块组成
        :param input_channels: 输入特征的通道数
        :param num_res_block: 子模块的残差块个数。
        :return:
        """
        layers = []
        for i in range(0, num_res_block):
            layers.append(ResBlock(in_channels=input_channels))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = self.conv(x)  # 只改变c, c: 3->32

        x = self.conv1(x)  # h,w/2, c: ->64
        x = self.layer1(x)  # 维度不变

        x = self.con2(x)  # h,w/4, c: ->128
        x = self.layer2(x)

        x = self.con3(x)  # h,w/8, c: ->256
        out3 = self.layer3(x)

        out4 = self.con4(out3)  # h,w/16, c: ->512
        out4 = self.layer4(out4)

        out5 = self.con5(out4)  # h,w/32, c: ->1024
        out5 = self.layer5(out5)

        return out3, out4, out5

