# -*- coding: utf-8 -*-
# @Time    : 2020/10/23 下午10:10
# @Author  : zxq
# @File    : yolov3_loss.py
# @Software: PyCharm

import math

import torch
import torch.nn as nn
import numpy as np

from utils.utils import bbox_iou


class YOLOLoss(nn.Module):
    def __init__(self, image_size, num_classes, anchors):
        super(YOLOLoss, self).__init__()
        self.image_size = image_size  # 原始图片大小: (x, y)
        self.num_classes = num_classes  # 检测目标类别数
        self.anchors = anchors  # [[x1, y1], [x2, y2], [x3, y3] 在原图上的尺度
        self.num_anchors = len(anchors)
        self.bbox_attrs = 5 + num_classes  # num_classes: 类别个数, bbox_attrs：属性个数。(x,y,w,h,conf,c0,c1,c2,...,c79)

        self.ignore_threshold = 0.5
        self.lambda_xy = 2.5
        self.lambda_wh = 2.5
        self.lambda_conf = 1.0
        self.lambda_cls = 1.0

        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, input, targets=None):
        """

        :param input: [b, c, h, w]
        :param targets: [b, num_gt, num_attr]. attr = [cls, x_ratio, y_ratio, w_ratio, h_ratio]. 存放的是比例, x_r = x/img_w
        :return:
        """
        batch_size = input.shape[0]
        in_h = input.shape[2]
        in_w = input.shape[3]
        stride_h = self.image_size[1] / in_h  # 高下采样的倍数
        stride_w = self.image_size[0] / in_w
        # 原图缩放了，anchor也要缩放对应的倍数，获取在特征图上的anchors
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]  # anchors缩放到对应的yolo输出层

        # [b,c,h,w] -> [b,num_anchors, bbox_attr,h,w] -> [b,num_anchors, h,w, bbox_attr]
        prediction = input.view(batch_size, self.num_anchors, self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4,
                                                                                                   2).contiguous()

        # Get outputs attr
        # [b,num_anchors,h,w,bbox_attr] -> [b, num_anchors,h,w]  (x,y)学的是中心坐标相对于cell左上角的偏移量 (0,1)之间
        x = torch.sigmoid(prediction[..., 0]).cuda()
        y = torch.sigmoid(prediction[..., 1]).cuda()  # -> [b, num_anchors,h,w]  Center y
        w = prediction[..., 2].cuda()  # -> [b, num_anchors,h,w]
        h = prediction[..., 3].cuda()  # -> [b, num_anchors,h,w]
        conf = torch.sigmoid(prediction[..., 4]).cuda()  # 目标概率
        pred_cls = torch.sigmoid(prediction[..., 5:].cuda())  # [b, num_anchors, h,w, num_classes]类别概率，bce loss

        # train
        if targets is not None:
            mask, noobj_mask, tx, ty, tw, th, tconf, tcls = \
                self.build_target(targets, scaled_anchors, in_w, in_h, self.ignore_threshold)

            mask, noobj_mask = mask.cuda(), noobj_mask.cuda()
            tx, ty, tw, th = tx.cuda(), ty.cuda(), tw.cuda(), th.cuda()
            tconf, tcls = tconf.cuda(), tcls.cuda()

            # loss
            # 1 location loss
            # x.shape: [b, num_anchors,h,w]. mask.shape: [b, num_anchors,h,w]
            loss_x = self.bce_loss(x * mask, tx * mask)  # x*mask: 预测的偏移量, tx: 标注的偏移量。mask值为1的位置是最佳anchor的位置
            loss_y = self.bce_loss(y * mask, ty * mask)
            loss_w = self.mse_loss(w * mask, tw * mask)
            loss_h = self.mse_loss(h * mask, th * mask)
            # 2 object loss
            # mask值为1的位置是有目标的cell，noobj_mask值为1的位置是没有目标的cell。
            loss_conf = self.bce_loss(conf * mask, mask) + 0.5 * self.bce_loss(conf * noobj_mask, noobj_mask * 0.0)
            # 3 class loss
            # pred_cls.shape: [2,3,52,52,80], mask.shape: [2,3,52,52]
            # 每个目标的类别信息都是一个80维向量，标注类别的对应维度的值是1，预测的值归一化到0-1之间
            loss_cls = self.bce_loss(pred_cls[mask == 1], tcls[mask == 1])  # pred_cls[mask == 1].shape: [num_obj, 80]

            #  total loss = losses * weight
            loss = (loss_x + loss_y) * self.lambda_xy + \
                   (loss_w + loss_h) * self.lambda_wh + \
                   loss_conf * self.lambda_conf + \
                   loss_cls * self.lambda_cls

            return loss, loss_x.item(), loss_y.item(), loss_w.item(), loss_h.item(), loss_conf.item(), loss_cls.item()

        # detect
        else:
            pass

    def build_target(self, target, anchors, in_w, in_h, ignore_threshold):
        """

        :param target: 标注的gt box信息. shape=[b, num_gt, num_attr]. attr = [cls, x_ratio, y_ratio, w_ratio, h_ratio].
        :param anchors: list. [(w1, h1), (w2, h2), (w3, h3)]. 在特征图尺度上的anchor
        :param in_w: 预测的特征图宽
        :param in_h: 预测的特征图高
        :param ignore_threshold: 计算标注的gt_bbox和3个anchor_box之间的iou，找到比较合适的anchor用于训练；
        比如长方形的目标，最好不要用竖直的anchor训练。每个cell有anchor，ignore_threshold值越大，忽略的anchor越多
        :return:
        noobj_mask: bool. noobj_mask[b, anchor_ious > ignore_threshold, gj, gi] = 0, 值为1的地方，没有目标

        以下记录的都是目标的最佳anchor信息
        mask: bool. mask[b, best_anchor_index, gj, gi] = 1. 值为1的地方，就是对应cell最佳的anchor
        tx: tx[b, best_anchor_index, gj, gi] = gx - gi  存放相对于cell(gj, gj)左上角的偏移量, 网络学习的是偏移量
        ty: ty[b, best_anchor_index, gj, gi] = gy - gj
        tw: tw[b, best_anchor_index, gj, gi] = math.log(gw / anchors[best_n][0] + 1e-16)，网络学习的是log(gw/aw)
        th: th[b, best_anchor_index, gj, gi] = math.log(gh / anchors[best_n][1] + 1e-16)
        tconf: tconf[b, best_n, gj, gi] = 1
        上面的shape都是：[b,3,f_h,f_w]
        tcls: tcls[b, best_n, gj, gi, int(target[b, t, 0])] = 1
        """

        batch_size = target.shape[0]

        mask = torch.zeros(batch_size, self.num_anchors, in_h, in_w,
                           requires_grad=False)  # [b,num_anchors,w,h]. [2,3,52,52]
        noobj_mask = torch.ones(batch_size, self.num_anchors, in_h, in_w, requires_grad=False)  # [b,num_anchors,w,h]
        tx = torch.zeros(batch_size, self.num_anchors, in_h, in_w, requires_grad=False)  # [b,num_anchors,w,h]
        ty = torch.zeros(batch_size, self.num_anchors, in_h, in_w, requires_grad=False)  # [b,num_anchors,w,h]
        tw = torch.zeros(batch_size, self.num_anchors, in_h, in_w, requires_grad=False)  # [b,num_anchors,w,h]
        th = torch.zeros(batch_size, self.num_anchors, in_h, in_w, requires_grad=False)  # [b,num_anchors,w,h]
        tconf = torch.zeros(batch_size, self.num_anchors, in_h, in_w, requires_grad=False)  # [b,num_anchors,w,h]
        # [b,num_anchors,w,h, num_cls]
        tcls = torch.zeros(batch_size, self.num_anchors, in_h, in_w, self.num_classes,
                           requires_grad=False)  # [2,3,52,52,80]
        for b in range(batch_size):  # 遍历batch中的每个图像
            for t in range(target.shape[1]):  # 遍历图像中的所有目标。target.shape=[b,num_obj,5]. 5=[cls, x_r,y_r,w_r,h_r]
                if target[b, t].sum() == 0:  # 当前图像中没有目标，每张图片的目标个数可能不同，组成batch时进行了填0操作
                    continue

                # 标注存放的x_ratio,y_ratio,w_ratio,h_ratio值是相对于原始图像的比例值,
                # 获取在特征图尺度下的gt标注bbox信息, target[b, t, 0]是类别
                gx = target[b, t, 1] * in_w  # float. 在特征层尺度的gt x坐标。tensor(0.3282) × 52 = 17.06
                gy = target[b, t, 2] * in_h  # tensor(0.7696) * 52 = 40.02
                gw = target[b, t, 3] * in_w  # 在特征层尺度上的高. tensor(0.4632) * 52 = 24.08
                gh = target[b, t, 4] * in_h  # 12.59
                # Get grid box indices
                # 17.06, 40.02 -> 17, 40
                gi = int(gx)  # 对特征图上的坐标gx向下取整
                gj = int(gy)  # (gi, gj)就是有目标的网格, 而(gx-gi,gy-gj)就是目标的中心坐标相对于网络左上角的偏移量

                # Get shape of gt box
                # tensor([ 0.0000,  0.0000, gw, gh]) -> tensor([[ 0.0000,  0.0000, gw, gh]])
                gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)  # tensor([[ 0.0000,  0.0000, 24.0841, 12.5948]])
                # Get shape of anchor box
                # ->(3, 4).  每一行是类似于[0. , 0. , 2.2, 3.4]的anchor宽高信息。
                anchor_box = torch.FloatTensor(np.concatenate((np.zeros((self.num_anchors, 2)),
                                                               np.array(anchors)), 1))
                # Calculate iou between gt and anchor shapes
                """
                gt_box = tensor([[0.0000, 0.0000, gw, gh]])
                anchor_box = tensor([[0.0000, 0.0000, 2.2000, 3.4000],
                                    [0.0000, 0.0000, 4.2000, 5.1000],
                                    [0.0000, 0.0000, 2.3000, 6.5000]])
                """
                anchor_ious = bbox_iou(gt_box, anchor_box)  # gt_box.shape: (1,4). anchor_box.shape: (3,4). ious.shape=3
                # Where the overlap is larger than threshold set mask to zero (ignore)
                # 大于阈值的是有目标，对应的3个anchors的位置设置为0。
                # noobj_mask值为1就没有目标，ignore_threshold越大，值为1的越多，忽略的anchor越多
                noobj_mask[b, anchor_ious > ignore_threshold, gj, gi] = 0
                # Find the best matching anchor box
                best_anchor_index = np.argmax(anchor_ious)

                # masks
                mask[b, best_anchor_index, gj, gi] = 1  # 最合适的anchor索引
                # Coordinates tx, ty
                tx[b, best_anchor_index, gj, gi] = gx - gi  # 存放相对于cell左上角的偏移量
                ty[b, best_anchor_index, gj, gi] = gy - gj
                # Width and height tw, th
                tw[b, best_anchor_index, gj, gi] = math.log(gw / anchors[best_anchor_index][0] + 1e-16)
                th[b, best_anchor_index, gj, gi] = math.log(gh / anchors[best_anchor_index][1] + 1e-16)
                # object
                tconf[b, best_anchor_index, gj, gi] = 1
                # One-hot encoding of label
                tcls[b, best_anchor_index, gj, gi, int(target[b, t, 0])] = 1

        return mask, noobj_mask, tx, ty, tw, th, tconf, tcls


if __name__ == '__main__':
    loss_module = YOLOLoss(image_size=(416, 416), num_classes=80, anchors=[[116, 90], [156, 198], [373, 326]])
    net_output = torch.rand(2, 255, 52, 52) * 10  # out5层的输出特征
    target1 = torch.FloatTensor([[16, 0.328250, 0.769577, 0.463156, 0.242207],
                                 [1, 0.128828, 0.375258, 0.249063, 0.733333],
                                 [0, 0.521430, 0.258251, 0.021172, 0.060869]])
    target2 = torch.FloatTensor([[59, 0.510930, 0.442073, 0.978141, 0.872188],
                                 [77, 0.858305, 0.073521, 0.074922, 0.059833],
                                 [0, 0.569492, 0.285235, 0.024547, 0.122254]])
    # [b, num_gt, num_attr]. [b, num_gt, cls, x_ratio, y_ratio, w_ratio, h_ratio]
    targets = torch.cat((target1.unsqueeze(0), target2.unsqueeze(0)), 0)  # [2, 2, 5]
    loss = loss_module(input=net_output, targets=targets)  # [b, num_gt, cls, x_r, y_r, w_r, h_r]