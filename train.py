import torch
import yaml

from yolov3 import YOLOv3

if __name__ == '__main__':
    cfg_dict = yaml.load(open('./config/cfg.yaml'), Loader=yaml.SafeLoader)
    yolo_module = YOLOv3(config=cfg_dict)
    x = torch.Tensor(4, 3, 416, 416)
    output3, output4, output5 = yolo_module(x)
    print(output3.shape, output4.shape, output5.shape)

    # YOLO loss with 3 scales
    yolo_loss = []

