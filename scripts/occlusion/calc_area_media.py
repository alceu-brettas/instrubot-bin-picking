#!/usr/bin/env python3

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

# Caminho do JSON gerado pela inferência
json_path = Path("experiments/C_yolo11_seg/infer/segmentations_test.json")

# Carrega o JSON
with open(json_path, 'r') as f:
    coco = json.load(f)

# Conta quantos objetos existem por imagem
image_ann_count = defaultdict(int)
for ann in coco['annotations']:
    image_ann_count[ann['image_id']] += 1

# Seleciona imagens que têm apenas 1 objeto (sem oclusão)
valid_image_ids = {img_id for img_id, count in image_ann_count.items() if count == 1}

# Coleta áreas por classe
areas_por_classe = defaultdict(list)
for ann in coco['annotations']:
    if ann['image_id'] in valid_image_ids:
        areas_por_classe[ann['category_id']].append(ann['area'])

# Calcula médias
area_media = {
    class_id: round(np.mean(areas), 2)
    for class_id, areas in areas_por_classe.items()
}

print("✅ Áreas médias por classe (para validação de oclusão):\n")
for k, v in area_media.items():
    print(f"Classe {k}: {v}")
