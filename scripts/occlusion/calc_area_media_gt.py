#!/usr/bin/env python3
"""
Calcula a área média de cada classe usando:
- o JSON anotado manualmente (ground truth) para selecionar imagens com apenas 1 objeto.
- o JSON de inferência (segmentations_test.json) para calcular áreas das máscaras inferidas.

Ideal para calibrar o algoritmo de oclusão.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

# === Caminhos ===
JSON_GT = Path("data/processed_seg/labels/test/coco.json")  # ground truth anotado (CVAT)
JSON_PRED = Path("experiments/C_yolo11_seg/infer/segmentations_test.json")  # saída da inferência

# === Carrega os arquivos JSON ===
with open(JSON_GT, "r") as f:
    gt = json.load(f)
with open(JSON_PRED, "r") as f:
    pred = json.load(f)

# === Identifica imagens com apenas 1 instrumento real ===
valid_image_ids = set()
gt_count = defaultdict(int)
for ann in gt["annotations"]:
    gt_count[ann["image_id"]] += 1
for img_id, count in gt_count.items():
    if count == 1:
        valid_image_ids.add(img_id)

# === Coleta áreas das máscaras inferidas nessas imagens ===
areas_por_classe = defaultdict(list)
for ann in pred["annotations"]:
    if ann["image_id"] in valid_image_ids:
        class_id = ann["category_id"]
        areas_por_classe[class_id].append(ann["area"])

# === Calcula a média por classe ===
area_media = {
    class_id: round(np.mean(areas), 2)
    for class_id, areas in areas_por_classe.items() if len(areas) > 0
}

# === Exibe resultado ===
print("\n✅ Áreas médias por classe (com base em imagens de GT com 1 objeto):\n")
for k, v in area_media.items():
    print(f"Classe {k}: {v}")
