#!/usr/bin/env python
# Treina YOLOv11 (task detect) usando apenas bounding-boxes

from ultralytics import YOLO
from pathlib import Path

DATA_YAML = "conf/data_mask.yaml"              # já criado
MODEL     = "weights/yolo11n-seg.pt"           # ou yolov11s.pt se houver GPU folgada
EPOCHS    = 100
IMG_SIZE  = 640
PROJECT   = Path("experiments/C_yolo11_seg")

def main():
    model = YOLO(MODEL)
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        project=str(PROJECT),
        name="run",
        exist_ok=True,
        batch=16,               # ajuste automático se faltar VRAM
        device="0",             # "0" GPU; "cpu" para CPU-only
        task="segment"          # MUITO IMPORTANTE
    )

if __name__ == "__main__":
    main()