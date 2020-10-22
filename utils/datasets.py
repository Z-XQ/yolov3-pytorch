import glob
import random
import os
import sys

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from utils.augmentations import horizontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms

"""
1.四维Tensor：传入四元素tuple(pad_l, pad_r, pad_t, pad_b)，
指的是（左填充，右填充，上填充，下填充），其数值代表填充次数
2.六维Tensor：传入六元素tuple(pleft, pright, ptop, pbottom, pfront, pback)，
指的是（左填充，右填充，上填充，下填充，前填充，后填充），其数值代表填充次数
"""


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)  # 如果是高小于宽，则上下填充(0, 0, pad1, pad2)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


"""
原图大小不是固定的，标注是


"""


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()  # list.

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]  # 获取label文件的路径, 原图片可以是png或者jpg结尾，但必须原图片的路径必须含有images
        self.img_size = img_size
        self.max_objects = 100  # ? 最多目标个数。好像没有用到
        self.augment = augment  # bool. 是否使用增强
        self.multiscale = multiscale  # bool. 是否多尺度输入，每次喂到网络中的batch中图片大小不固定。
        self.normalized_labels = normalized_labels  # bool. 默认label.txt文件中的bbox是归一化到0-1之间的
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0  # 当前网络训练的是第几个batch

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()  # 如果index超过图像总数，则取整

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)  # 第一个维度是留给batch size
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)  # 如果标注bbox不是归一化的，则标注里面的保存的就是真实位置
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)  # 搞成 w=h
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()  # 和获取image同样方式
        print(label_path)

        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))  # [0class_id, 1x_c, 2y_c, 3w, 4h] 归一化的
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)  # 使用(x_c, y_c, w, h)获取真实坐标（左上，右下）
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # Adjust for added padding 标注要和原图做相同的调整 pad（0左，1右，2上，3下）
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Returns (x, y, w, h) 标注有xyxy,转成xywh(其中xy是中心坐标，且被归一化)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w  # (padded_w, padded_h)是当前padding之后图片的宽度
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            # boxes[:, 3] *= w_factor / padded_w
            # boxes[:, 4] *= h_factor / padded_h

            # (w_factor, h_factor)是原始图的宽高
            boxes[:, 3] = boxes[:, 3] * w_factor / padded_w  # boxes[:, 3] * w_factor获取真实bbox宽度，再除以padded_w归一化
            boxes[:, 4] = boxes[:, 4] * h_factor / padded_h  # 0.1 × 480 / 640 或者是 48 x 1 / 640

            targets = torch.zeros((len(boxes), 6))  # [4, 6]
            targets[:, 1:] = boxes  # [[ 0.0000, 16.0000,  0.5337,  0.6046,  0.3430,  0.2945], ...,])

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horizontal_flip(img, targets)  # 只有一种增强！

        return img_path, img, targets  # 原图图片路径，padding之后的图片，对应的标注normalized bbox (img_id, class_id, x_c, y_c, w, h)

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))  # imgs每张图片大小不一定相同
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i  # boxes第0维记录图片的id
        targets = torch.cat(targets, 0)  # [104, 6], 直接将一个batch中所有的bbox合并在一起，计算loss时是按batch计算
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape

        # 这里并没有调整标注，因为标注是图片宽高的相对大小，缩放原图后，这个比例不会发生改变
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])  # 训练的时候，每10个batch变换一次图片大小
        self.batch_count += 1
        return paths, imgs, targets  # [img_id, class_id, x_c, y_c, h, w]

    def __len__(self):
        return len(self.img_files)
