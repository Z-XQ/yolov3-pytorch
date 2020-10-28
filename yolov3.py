# -*- coding: utf-8 -*-
# @Time    : 2020/10/22 下午10:10
# @Author  : zxq
# @File    : yolov3.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from backbone.darknet53 import Conv2dBatchLeaky, darknet53


class Conv2dBlock5L(nn.Module):
    """
    对应网络结构图中的Conv2D Block 5L，具体功能是6个conv+bn+leakyReLU，
    为什么叫5L，我猜是输出通道有5次是在c1和c2两种之间变换
    只改变通道数
    """

    def __init__(self, in_channels, out_channels):
        """
        :param in_channels: 前面DarkNet输出的特征图通道数
        :param out_channels: list. [c1, c2]. 通道数就在c1和c2之间变化，最后输出c2通道数
        然后5个卷积的通道数就在in_channels和in_channels//2两者间变化
        """
        super(Conv2dBlock5L, self).__init__()
        conv = Conv2dBatchLeaky(in_channels=in_channels, out_channels=out_channels[0], kernel_size=1, stride=1)  # 降维，减少计算量

        conv1 = Conv2dBatchLeaky(in_channels=out_channels[0], out_channels=out_channels[1], kernel_size=3, stride=1)
        conv2 = Conv2dBatchLeaky(in_channels=out_channels[1], out_channels=out_channels[0], kernel_size=1, stride=1)
        conv3 = Conv2dBatchLeaky(in_channels=out_channels[0], out_channels=out_channels[1], kernel_size=3, stride=1)
        conv4 = Conv2dBatchLeaky(in_channels=out_channels[1], out_channels=out_channels[0], kernel_size=1, stride=1)
        conv5 = Conv2dBatchLeaky(in_channels=out_channels[0], out_channels=out_channels[1], kernel_size=3, stride=1)
        self.out_channels = out_channels[1]

        # 打包下，省得在forward重复写
        self.layers = nn.Sequential(
            conv,
            conv1,
            conv2,
            conv3,
            conv4,
            conv5
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class YOLOv3(nn.Module):
    def __init__(self, config):
        super(YOLOv3, self).__init__()
        self.backbone = darknet53(pretrained=False)
        # num_anchors * (5+num_classes): 3 * (5+ 80) = 255
        anchors = config['yolo']['anchor']  # [10,13,  16,30,  33,23,  30,61,  62,45, ...]  # 9个
        self.anchors = [(anchors[i], anchors[i + 1]) for i in
                        range(0, len(anchors) - 1, 2)]  # [(10,13),  (16,30),  ...]
        num_anchors = len(self.anchors) // 3  # 平均分成3份
        num_classes = config['yolo']['classes']

        # 默认每个输出层的anchor个数都是len(config['yolo']['anchor'][0]),
        # 对于每个输出层的所有位置输出属性维度： coco: 3x85=255, 图中是3x(5+20)=75
        self.final_out_channels = num_anchors * (5 + num_classes)

        # 1, stride 32
        # output_channels[-1]是DarkNet最后一层输出, 这里layer5对应的尺度是DarkNet第5个模块的输出尺度
        self.block_layer5 = Conv2dBlock5L(in_channels=self.backbone.output_channels[-1], out_channels=[512, 1024])
        # yolo layer，这里使用1x1卷积，简单的把channels修改为self.final_out_channels
        self.conv1x1_out5 = nn.Conv2d(in_channels=self.block_layer5.out_channels, out_channels=self.final_out_channels,
                                      kernel_size=1, stride=1, padding=0, bias=True)

        # 2, stride 16
        # 对应结构图中的Conv2D + UpSampling2D, 其中conv用来修改通道数，upsample用来修改高宽尺度
        # channels: -> 256
        self.conv5 = Conv2dBatchLeaky(in_channels=self.block_layer5.out_channels, out_channels=256, kernel_size=1,
                                      stride=1)
        # upSample: 13x13 -> 26x26
        self.up_sample = Upsample(scale_factor=2, mode='nearest')
        # concat up_sample4 + backbone.out4
        in_channels = self.backbone.output_channels[-2] + 256  # 512+256=768
        # yolo layer 4
        self.block_layer4 = Conv2dBlock5L(in_channels=in_channels, out_channels=[256, 512])  # 768->512
        self.conv1x1_out4 = nn.Conv2d(in_channels=self.block_layer4.out_channels, out_channels=self.final_out_channels,
                                      kernel_size=1, stride=1, padding=0, bias=True)

        # 3, stride 8
        self.conv4 = Conv2dBatchLeaky(in_channels=self.block_layer4.out_channels, out_channels=128, kernel_size=1,
                                      stride=1)  # 512 -> 128
        # up_sample3: 26x26 -> 52x52
        # concat: up_sample3 + backbone.out3
        in_channels = self.backbone.output_channels[-3] + 128  # 256+128=384
        # yolo layer 3
        self.block_layer3 = Conv2dBlock5L(in_channels=in_channels, out_channels=[128, 256])  # channels: -> 256
        self.conv1x1_out3 = nn.Conv2d(in_channels=self.block_layer3.out_channels, out_channels=self.final_out_channels,
                                      kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        backbone_out3, backbone_out4, backbone_out5 = self.backbone(x)  # [b,256,52,52],[b,512,26,26],[b,1024,52,52]

        # stride 32
        block_out5 = self.block_layer5(backbone_out5)  # [b,1024,13,13]. chw都没变，1024,13,13
        yolo_out5 = self.conv1x1_out5(block_out5)  # [b,1024,13,13]->[b,255,13,13]省去了一步conv3x3，这里通过1x1的卷积输出固定channel的特征图

        # stride 16
        x = self.conv5(block_out5)  # [b,1024,13,13] -> [b,256,13,13]
        x = self.up_sample(x)  # [b,256,13,13] -> [b,256,26,26]
        x = torch.cat([backbone_out4, x], 1)  # backbone_out4: [b,512,26,26], x: [b,256,26,26] -> [b,768,26,26]
        block_out4 = self.block_layer4(x)  # [b,768,26,26] -> [b,512,26,26], 图中是变成[256]
        yolo_out4 = self.conv1x1_out4(block_out4)  # [b,512,26,26] -> [b,255,26,26]

        # stride 8
        x = self.conv4(block_out4)  # [b,512,26,26] -> [b,128,26,26]
        x = self.up_sample(x)  # [b,128,26,26] -> [b,128,52,52]
        x = torch.cat([backbone_out3, x], 1)  # backbone_out3: [b,256,52,52], x: [b,128,52,52] -> [b,384,52,52]
        block_out3 = self.block_layer3(x)  # [b,384,52,52] -> [b,256,52,52]
        yolo_out3 = self.conv1x1_out3(block_out3)  # [b,256,52,52] -> [b,255,52,52]

        return yolo_out3, yolo_out4, yolo_out5


if __name__ == '__main__':
    cfg_dict = yaml.load(open('./config/cfg.yaml'), Loader=yaml.SafeLoader)
    yolo_module = YOLOv3(config=cfg_dict)
    x = torch.Tensor(4, 3, 416, 416)
    output3, output4, output5 = yolo_module(x)
    print(output3.shape, output4.shape, output5.shape)
