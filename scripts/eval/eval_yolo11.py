#!/usr/bin/env python3
"""
Avaliação objetiva (mAP, IoU, FPS) para YOLO-v11.
Necessita:  ultralytics  |  torch  |  py-cocotools
"""

import argparse, csv, pathlib, time
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="caminho p/ best.pt")
    ap.add_argument("--data",  default="conf/data_bbox.yaml")
    ap.add_argument("--split", default="test", choices=["val","test"])
    ap.add_argument("--device",default="0")          # "cpu" ou "0,1"
    args = ap.parse_args()

    model = YOLO(args.model)
    t0 = time.time()
    metrics = model.val(data=args.data, split=args.split, device=args.device, plots=False, save_json=False)
    dur = time.time() - t0

    res = {
        "model": pathlib.Path(args.model).name,
        "split": args.split,
        "mAP50": metrics.box.map50,
        "mAP5095": metrics.box.map,
        "precision": metrics.box.precision,
        "recall": metrics.box.recall,
        "fps": metrics.speed['inference'] and 1000/metrics.speed['inference'] or 0,
        "time_s": round(dur,1)
    }
    print("\n=== Resultado ===")
    for k,v in res.items(): print(f"{k:10}: {v}")

    out_csv = pathlib.Path("runs/eval/summary.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_csv.exists()
    with out_csv.open("a", newline="") as cf:
        w = csv.DictWriter(cf, fieldnames=res.keys())
        if write_header: w.writeheader()
        w.writerow(res)

if __name__ == "__main__":
    main()
