import torch
import torch.nn as nn
import numpy as np

from utils.utils import bbox_iou


class YOLOLoss(nn.Module):
    def __init__(self, image_size, num_classes, anchors):
        self.image_size = image_size  # 原始图片大小: (x, y)
        self.anchors = anchors  # [[x1, y1], [x2, y2], [x3, y3] 在原图上的尺度
        self.num_anchors = len(anchors)
        self.bbox_attrs = 5 + num_classes  # num_classes: 类别个数, bbox_attrs：属性个数。(x,y,w,h,conf,c0,c1,c2,...,c79)
        self.ignore_threshold = 0.5

        pass

    def forward(self, input, targets=None):
        """

        :param input: [b, c, h, w]
        :param targets: 7. [b, num_gt, cls, x_r, y_r, w_r, h_r]
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

        # attr
        x = prediction[..., 0]
        y = prediction[..., 1]
        w = prediction[..., 2]
        h = prediction[..., 3]
        conf = torch.sigmoid(prediction[..., 4])  # 目标概率
        pred_cls = prediction[..., 5:]  # 类别概率

        #
        if targets is not None:
            self.build_target(targets, scaled_anchors, in_w, in_h, self.ignore_threshold)

    def build_target(self, target, anchors, in_w, in_h, ignore_threshold):
        """

        :param target: 7. [b, num_gt, cls, x_ratio, y_ratio, w_ratio, h_ratio]. 标注的gt box信息
        :param anchors: list. [(w1, h1), (w2, h2), (w3, h3)]. 在特征图尺度上的anchor
        :param in_w: 预测的特征图宽
        :param in_h: 预测的特征图高
        :param ignore_threshold: 计算标注的gt_bbox和3个anchor_box之间的iou，找到比较合适的anchor用于训练；
        长方形的目标，最好不要用竖直的anchor训练。
        :return:
        """

        batch_size = target.shape[0]

        mask = torch.zeros(batch_size, self.num_anchors, in_h, in_w, requires_grad=False)  # [b,num_anchors,w,h]
        noobj_mask = torch.ones(batch_size, self.num_anchors, in_h, in_w, requires_grad=False)  # [b,num_anchors,w,h]
        tx = torch.zeros(batch_size, self.num_anchors, in_h, in_w, requires_grad=False)  # [b,num_anchors,w,h]
        ty = torch.zeros(batch_size, self.num_anchors, in_h, in_w, requires_grad=False)  # [b,num_anchors,w,h]
        tw = torch.zeros(batch_size, self.num_anchors, in_h, in_w, requires_grad=False)  # [b,num_anchors,w,h]
        th = torch.zeros(batch_size, self.num_anchors, in_h, in_w, requires_grad=False)  # [b,num_anchors,w,h]
        tconf = torch.zeros(batch_size, self.num_anchors, in_h, in_w, requires_grad=False)  # [b,num_anchors,w,h]
        tcls = torch.zeros(batch_size, self.num_anchors, in_h, in_w, requires_grad=False)  # [b,num_anchors,w,h]
        for b in range(batch_size):  # 遍历batch中的每个图像
            for t in range(target.shape[1]):  # 遍历图像中的所有目标
                if target[b, t].sum() == 0:  # 当前图像中没有目标
                    continue

                # 标注存放的x_ratio,y_ratio,w_ratio,h_ratio值是相对于原始图像的比例值,
                # 获取在特征图尺度下的gt标注bbox信息
                gx = target[b, t, 1] * in_w  # float. 获取yolo某一层的gt x坐标
                gy = target[b, t, 2] * in_h
                gw = target[b, t, 3] * in_w  # 在特征层尺度上的高
                gh = target[b, t, 4] * in_h
                # Get grid box indices
                gi = int(gx)  # 对特征图上的坐标gx取整
                gj = int(gy)  # (gi, gj)就是有目标的网格

                # Get shape of gt box
                # tensor([ 0.0000,  0.0000, gw, gh]) -> tensor([[ 0.0000,  0.0000, gw, gh]])
                gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
                # Get shape of anchor box
                """
                np.zeros((self.num_anchors, 2)).  ->(3,2)
                a = array([[0., 0.],
                           [0., 0.],
                           [0., 0.]])
                np.array(anchors)).   ->(3, 2) 
                b = array([[2.2, 3.4],
                           [4.2, 5.1],
                           [2.3, 6.5]])
                np.concatenate((a, b), 1).   ->(3, 4)
                array([[0. , 0. , 2.2, 3.4],
                       [0. , 0. , 4.2, 5.1],
                       [0. , 0. , 2.3, 6.5]])
                    
                """
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
                anchor_ious = bbox_iou(gt_box, anchor_box)  # gt_box.shape: (1,4). anchor_box.shape: (3,4)
                # Where the overlap is larger than threshold set mask to zero (ignore)
                noobj_mask[b, anchor_ious > ignore_threshold, gj, gi] = 0
                # Find the best matching anchor box
                best_anchor_index = np.argmax(anchor_ious)

                # masks
                mask[b, best_anchor_index, gj, gi] = 1  # 最合适的anchor索引
                tx[b, best_anchor_index, gj, gi] =
