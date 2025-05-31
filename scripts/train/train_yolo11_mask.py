#!/usr/bin/env python
# Treina YOLOv11 em modo "segment" com caixas + máscaras COCO

from ultralytics import YOLO
from pathlib import Path

DATA_YAML = "conf/data_mask.yaml"       # inclui mask_path
MODEL     = "yolov11n-seg.pt"           # versão de segmentação
EPOCHS    = 120
IMG_SIZE  = 640
PROJECT   = Path("experiments/C_yolo11_mask")

def main():
    model = YOLO(MODEL)
    model.train(
        data=DATA_YAML,
        task="segment",     
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        project=str(PROJECT),
        name="run",
        exist_ok=True,
        batch=8,              # segment consome +VRAM
        device=0
    )

if __name__ == "__main__":
    main()
