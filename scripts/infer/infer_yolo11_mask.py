#!/usr/bin/env python3
"""
Detecção + máscaras SAM + heurística de oclusão.
Pinta de VERDE o objeto considerado "em cima" (menor nº de blobs).
"""

import argparse, cv2, json, numpy as np, pathlib
from ultralytics import YOLO, SAM
from tqdm import tqdm

sam = SAM('sam_vit_h')

def mask_from_box(image, box):
    """box xyxy → binary mask bool(H,W)."""
    masks = sam(image, [box], multimask_output=False)
    return masks[0]['mask'].astype(np.uint8)

def decide_top(masks):
    """Retorna índice da máscara com menos blobs conexos."""
    blob_counts = [cv2.connectedComponents(m)[0]-1 for m in masks]
    return int(np.argmin(blob_counts))

def overlay(image, masks, top_idx):
    over = image.copy()
    for i,m in enumerate(masks):
        color = (0,255,0) if i==top_idx else (0,0,255)
        over[m.astype(bool)] = 0.4*over[m.astype(bool)] + 0.6*np.array(color)
    return over

# ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",  required=True)
    ap.add_argument("--source", default="data/raw/test")
    ap.add_argument("--outdir", default="runs/infer/mask")
    ap.add_argument("--conf",   type=float, default=0.25)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    model = YOLO(args.model)
    src, outdir = pathlib.Path(args.source), pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(sorted(src.glob("*.*g"))):
        img = cv2.imread(str(img_path))
        result = model(img, device=args.device, conf=args.conf, verbose=False)[0]

        boxes = result.boxes.xyxy.cpu().numpy()
        masks = [mask_from_box(img, box) for box in boxes] if len(boxes) else []
        top_idx = decide_top(masks) if len(masks) >= 2 else 0

        vis = overlay(img, masks, top_idx) if masks else img
        cv2.imwrite(str(outdir / img_path.name), vis)

        meta = {"boxes": boxes.tolist(),
                "top_idx": int(top_idx)}
        (outdir / (img_path.stem + ".json")).write_text(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
