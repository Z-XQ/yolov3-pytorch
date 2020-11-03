import logging
import os
import sys

import torch
import yaml
import torch.optim as optim
from torch.utils.data import DataLoader

MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))

from utils.coco_dataset import COCODataset
from yolov3_loss import YOLOLoss
from yolov3_module import YOLOv3


def main():
    logging.basicConfig(level=logging.DEBUG, format='[%()]')


if __name__ == '__main__':
    cfg_dict = yaml.load(open('./config/cfg.yaml'), Loader=yaml.SafeLoader)

    # Load and initialize network
    yolo_module = YOLOv3(config=cfg_dict)
    yolo_module.train()  # 开启训练模式
    yolo_module.cuda()

    # Optimizer and learning rate
    optimizer = optim.SGD(yolo_module.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # YOLO loss with 3 scales
    yolo_loss = []
    anchors = yolo_module.anchors
    for i in range(3):
        one_loss_module = YOLOLoss(image_size=cfg_dict['img_size'], num_classes=cfg_dict['num_classes'],
                                   anchors=anchors[i:i + 3])
        yolo_loss.append(one_loss_module)

    # DataLoader
    DataLoader(dataset=COCODataset(list_path=cfg_dict['train_path'], img_size=cfg_dict['img_size'], is_training=True),
               batch_size=cfg_dict['batch_size'],
               shuffle=True,
               num_workers=32, pin_memory=True)

    # logging.info("Start training.")

