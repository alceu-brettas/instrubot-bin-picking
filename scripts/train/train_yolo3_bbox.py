# scripts/train/train_yolo3_bbox.py

import sys
from pathlib import Path

# Ajusta o path para importar YOLOv3 localmente
ROOT = Path(__file__).resolve().parents[2]
YOLOV3_PATH = ROOT / "yolov3"
sys.path.append(str(YOLOV3_PATH))

from train import train
from utils.torch_utils import select_device
from utils.callbacks import Callbacks

def main():
    opt = {
        'weights': 'yolov3.pt',
        'cfg': 'cfg/yolov3.yaml',
        'data': 'conf/data_bbox.yaml',
        'hyp': 'data/hyp.scratch.yaml',
        'epochs': 100,
        'batch_size': 16,
        'imgsz': 640,
        'device': 'cpu',
        'project': 'experiments/A_yolo3_bbox',
        'name': 'run',
        'exist_ok': True,
        'rect': False,
    }

    device = select_device(opt['device'])
    callbacks = Callbacks()

    train(opt, device, callbacks, wandb=None)

if __name__ == '__main__':
    main()
