import cv2
import numpy as np
from pathlib import Path

# Caminhos
IMAGE_DIR = Path("data/processed_det/images/test")
LABEL_BBOX_DIR = Path("data/processed_det/labels/test")
LABEL_MASK_DIR = Path("data/processed_seg/labels/test")
OUTPUT_DIR = Path("experiments/gt_viz_seg")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Transparência das máscaras (0 = transparente, 1 = opaco)
MASK_OPACITY = 0.4


# Lista de nomes das classes
class_names = {
    0: "1-Bisturi",
    1: "2-Tesoura_Curva",
    2: "3-Tesoura_Reta",
    3: "4-Pinca"
}

# Cores por classe (BGR)
class_colors = {
    0: (255, 0, 0),
    1: (255, 255, 0),
    2: (255, 255, 255),
    3: (125, 255, 50),
}

def draw_polygons(image, polygons, color, alpha=MASK_OPACITY):
    """
    Desenha uma lista de polígonos sobre a imagem com cor e opacidade especificadas.
    """
    overlay = image.copy()
    for poly in polygons:
        cv2.fillPoly(overlay, [poly], color)
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

def parse_coco_polygons(file_path, img_width, img_height):
    """
    Lê um arquivo .txt estilo COCO (YOLOv5-seg) e retorna uma lista de (cls_id, polygon)
    """
    polygons = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) < 7:  # cls_id + pelo menos 3 pontos (6 valores) => 7+
                continue
            cls_id = int(parts[0])
            points = list(map(float, parts[1:]))

            pts = np.array([
                [int(x * img_width), int(y * img_height)]
                for x, y in zip(points[::2], points[1::2])
            ], dtype=np.int32)

            polygons.append((cls_id, pts))
    return polygons

def draw_gt_boxes_and_masks(image_path, label_bbox_path, label_mask_path, output_path):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Imagem não encontrada: {image_path}")
        return

    height, width = image.shape[:2]

    # === DESENHAR MÁSCARAS ===
    if label_mask_path.exists():
        mask_polygons = parse_coco_polygons(label_mask_path, width, height)
        for cls_id, poly in mask_polygons:
            color = class_colors.get(cls_id, (0, 255, 0))
            image = draw_polygons(image, [poly], color, alpha=0.4)
    else:
        print(f"⚠️ Sem máscara: {label_mask_path.name}")

    # === DESENHAR BOUNDING BOXES ===
    if label_bbox_path.exists():
        with open(label_bbox_path, 'r') as f:
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

                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(image, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
                cv2.putText(image, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (35, 35, 35), 1)
    else:
        print(f"⚠️ Sem label de caixa: {label_bbox_path.name}")

    cv2.imwrite(str(output_path), image)

# Processa todas as imagens
for image_path in IMAGE_DIR.glob("*.jpeg"):
    name = image_path.stem
    label_bbox_path = LABEL_BBOX_DIR / f"{name}.txt"
    label_mask_path = LABEL_MASK_DIR / f"{name}.txt"
    output_path = OUTPUT_DIR / f"{name}.jpeg"
    draw_gt_boxes_and_masks(image_path, label_bbox_path, label_mask_path, output_path)

print(f"✅ Imagens com GT e máscaras salvas em: {OUTPUT_DIR}")
