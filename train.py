import logging
import os
import sys
import time

import torch
import yaml
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))

from utils.coco_dataset import COCODataset
from yolov3_loss import YOLOLoss
from yolov3_module import YOLOv3


def _save_checkpoint(state_dict, config):
    # global best_eval_result
    checkpoint_path = os.path.join(config["sub_working_dir"], "model.pth")
    torch.save(state_dict, checkpoint_path)
    logging.info("Model checkpoint saved to %s" % checkpoint_path)


def main(cfg_dict):
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
    data_loader = DataLoader(
        dataset=COCODataset(list_path=cfg_dict['train_path'], img_size=cfg_dict['img_size'], is_training=True),
        batch_size=cfg_dict['batch_size'],
        shuffle=True,
        num_workers=32, pin_memory=True)

    # Start the training loop
    logging.info("Start training.")
    global_step = 0
    for epoch in range(cfg_dict['epochs']):
        for step, samples in enumerate(data_loader):
            images, labels = samples['image'], samples['label']
            start_time = time.time()
            global_step += 1

            # Forward and backward
            optimizer.zero_grad()
            outputs = yolo_module(images)

            # cal loss
            losses_name = ["total loss", "x", "y", "w", "h", "conf", "cls"]
            losses = []  # [[], [], [], ...]  # 第一个存的都是total loss
            [losses.append([]) for i in range(len(losses_name))]
            for i in range(3):
                one_scale_loss = yolo_loss[i](input=outputs[i], targets=labels)
                for j, l in enumerate(one_scale_loss):  # 遍历["total loss", "x", "y", "w", "h", "conf", "cls"]
                    losses[j].append(l)  # losses的第一个元素存了三个total loss
            losses = [sum(l) for l in losses]  # 对x, y, w等不同的loss分别求和
            loss = losses[0]  # total loss

            # backward
            loss.backward()
            optimizer.step()

            # tensorboard
            tb_writer = SummaryWriter(log_dir=cfg_dict['log_dir'])
            if step > 0 and step % 10 == 0:
                _loss = loss.item()
                duration = float(time.time() - start_time)
                example_per_second = cfg_dict["batch_size"] / duration
                lr = optimizer.param_groups[0]['lr']
                logging.info(
                    "epoch [%.3d] iter = %d loss = %.2f example/sec = %.3f lr = %.5f " %
                    (epoch, step, _loss, example_per_second, lr)
                )
                tb_writer.add_scalar("lr", lr, global_step)
                tb_writer.add_scalar("example/sec", example_per_second, global_step)
                for i, name in enumerate(losses_name):
                    value = _loss if i == 0 else losses[i]
                    tb_writer.add_scalar(name, value, global_step)

            if step > 0 and step % 1000 == 0:
                # net.train(False)
                _save_checkpoint(YOLOv3.state_dict(), cfg_dict)


if __name__ == '__main__':
    cfg = yaml.load(open('./config/cfg.yaml'), Loader=yaml.SafeLoader)
    main(cfg)




