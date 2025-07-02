import cv2
import os
from pathlib import Path

# Caminhos
IMAGE_DIR = Path("data/processed_det/images/test")
LABEL_DIR = Path("data/processed_det/labels/test")
OUTPUT_DIR = Path("experiments/gt_viz")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Lista de nomes das classes
class_names = {
    0: "1-Bisturi",
    1: "2-Tesoura_Curva",
    2: "3-Tesoura_Reta",
    3: "4-Pinca"
}

# Cores por classe (BGR) - estilo Ultralytics
class_colors = {
    0: (255, 0, 0),     # Azul
    1: (255, 255, 0),   # Ciano claro
    2: (255, 255, 255), # Branco
    3: (125, 255, 50),   # Amarelo claro / Verde limão
}

def draw_gt_boxes(image_path, label_path, output_path):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Imagem não encontrada: {image_path}")
        return

    height, width = image.shape[:2]

    if not label_path.exists():
        print(f"⚠️ Sem label: {label_path.name}")
        return

    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls_id, x_center, y_center, w, h = map(float, parts)
            cls_id = int(cls_id)

            x1 = int((x_center - w / 2) * width)
            y1 = int((y_center - h / 2) * height)
            x2 = int((x_center + w / 2) * width)
            y2 = int((y_center + h / 2) * height)

            label = class_names.get(cls_id, str(cls_id))
            color = class_colors.get(cls_id, (0, 255, 0))

            # Desenha retângulo
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Label com fundo (estilo Ultralytics)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (35, 35, 35), 1)

    cv2.imwrite(str(output_path), image)

# Processa todas as imagens
for image_path in IMAGE_DIR.glob("*.jpeg"):
    label_path = LABEL_DIR / (image_path.stem + ".txt")
    output_path = OUTPUT_DIR / image_path.name
    draw_gt_boxes(image_path, label_path, output_path)

print(f"✅ Imagens com GT visualizado salvas em: {OUTPUT_DIR}")
