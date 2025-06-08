#!/usr/bin/env python3
"""
Inferência com YOLOv11 (segmentação) e exportação COCO-style JSON.
Gera segmentações em lote para o conjunto de teste.
"""

from pathlib import Path
import json
import cv2
from ultralytics import YOLO

# 2147
# 2043
# 1469 (sem oclusao mas imagem boa)

# 2759
# 1223
# 2763
# 2048

# === Parâmetros ===
MODEL_PATH = "experiments/C_yolo11_seg/run/weights/best.pt"
SOURCE_PATH = Path("data/processed_seg/images/test")
OUTPUT_JSON = Path("experiments/C_yolo11_seg/infer/segmentations_test.json")
CONFIDENCE_THRESHOLD = 0.25

# === Inicializa modelo ===
model = YOLO(MODEL_PATH)

# === Executa inferência ===
results = model.predict(
    source=str(SOURCE_PATH),
    imgsz=640,
    conf=CONFIDENCE_THRESHOLD,
    device="cpu",
    save=True,
    project="experiments/C_yolo11_seg/infer",
    name="images_with_masks",
    exist_ok=True
)

# === Estrutura COCO ===
annotations = []
images = []
ann_id = 1
category_map = {
    0: "Bisturi",
    1: "Tesoura Curva",
    2: "Tesoura Reta",
    3: "Pinca"
}

for result in results:
    filename = Path(result.path).name
    image_id = int(Path(result.path).stem.lstrip("0") or "0")  # Ex: "00025.jpg" → 25

    image_info = {
        "id": image_id,
        "file_name": filename,
        "width": result.orig_shape[1],
        "height": result.orig_shape[0]
    }
    images.append(image_info)

    if result.masks is not None:
        for i, seg in enumerate(result.masks.xy):
            segmentation = [seg.flatten().tolist()]
            annotation = {
                "id": ann_id,
                "image_id": image_id,
                "category_id": int(result.boxes.cls[i]),
                "segmentation": segmentation,
                "iscrowd": 0,
                "area": float(cv2.contourArea(seg.astype('float32'))),
                "bbox": list(map(float, result.boxes.xywh[i])),
            }
            annotations.append(annotation)
            ann_id += 1

# === Exporta JSON ===
coco_output = {
    "images": images,
    "annotations": annotations,
    "categories": [
        {"id": 0, "name": "Bisturi"},
        {"id": 1, "name": "Tesoura Curva"},
        {"id": 2, "name": "Tesoura Reta"},
        {"id": 3, "name": "Pinca"}
    ]
}

OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_JSON, "w") as f:
    json.dump(coco_output, f, indent=2)

print(f"\n✅ JSON COCO salvo em: {OUTPUT_JSON}")
