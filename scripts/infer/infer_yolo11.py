#!/usr/bin/env python3
"""
Detecção em lote (BBox) com YOLO-v11.
Salva .jpg anotado e .json de outputs.
"""

import argparse, cv2, json, pathlib
from ultralytics import YOLO
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",  required=True)
    ap.add_argument("--source", default="data/raw/test")
    ap.add_argument("--outdir", default="runs/infer/bbox")
    ap.add_argument("--conf",   type=float, default=0.25)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    model = YOLO(args.model)
    src = pathlib.Path(args.source)
    outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(sorted(src.glob("*.*g"))):
        result = model(img_path, device=args.device, conf=args.conf, verbose=False)[0]
        im_annot = result.plot()                       # caixas desenhadas
        cv2.imwrite(str(outdir / img_path.name), im_annot)

        # salva json raso
        j = [{"cls": int(b.boxes.cls[i]),
              "conf": float(b.boxes.conf[i]),
              "xyxy": [float(v) for v in b.boxes.xyxy[i]]}
             for i,b in enumerate(result)]
        (outdir / (img_path.stem + ".json")).write_text(json.dumps(j, indent=2))

if __name__ == "__main__":
    main()
