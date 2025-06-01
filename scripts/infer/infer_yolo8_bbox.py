# scripts/infer/infer_yolo11_bbox.py

from ultralytics import YOLO
from pathlib import Path
import sys

MODEL_PATH = "experiments/D_yolo8_bbox/run/weights/my_yolo_model.pt"
DEFAULT_SOURCE = "data/processed/images/test"  # pasta test como fallback
SAVE_PROJECT = "experiments/D_yolo8_bbox/run/infer"
SAVE_NAME = "infer"

def infer(source_path):
    model = YOLO(MODEL_PATH)

    results = model.predict(
        source=source_path,
        imgsz=640,
        conf=0.25,
        device="cpu",
        save=True,
        show=is_single_image(source_path),
        project=SAVE_PROJECT,
        name=SAVE_NAME,
        exist_ok=False
    )

    print(f"\n✅ Resultado salvo em: {results[0].save_dir}\n")

def is_single_image(path_str):
    path = Path(path_str)
    return path.is_file() and path.suffix.lower() in [".jpg", ".jpeg", ".png"]

def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = DEFAULT_SOURCE

    if not Path(path).exists():
        print(f"❌ Caminho não encontrado: {path}")
        return

    infer(path)

if __name__ == "__main__":
    main()
