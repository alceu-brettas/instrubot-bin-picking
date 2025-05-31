#!/usr/bin/env python3
"""
infer_yolov2_darknet.py
-----------------------
• Carrega libdarknet.so em Python.
• Faz inferência em todas as imagens de uma pasta.
• Usa SAM (opcional) p/ gerar máscaras; decide qual objeto está em cima.
• Salva .jpg com boxes + highlight verde no objeto top.
"""

import argparse, cv2, numpy as np, pathlib, json, torch
from ultralytics import SAM
import darknet  # libdarknet.so deve estar no PYTHONPATH

ROOT = pathlib.Path(__file__).resolve().parents[1]
sam_model = SAM('sam_vit_h')  # baixe/pesquise caminho conforme seu env

# ---------- Funções utilitárias ---------------------------------------------
def load_net(cfg, weights):
    net = darknet.load_net_custom(
        str(cfg).encode(), str(weights).encode(), 0, 1)
    meta = darknet.load_meta(str(ROOT / "conf/classes.names").encode())
    return net, meta

def detect(net, meta, im):
    darknet_image = darknet.make_image(im.shape[1], im.shape[0], 3)
    darknet.copy_image_from_bytes(darknet_image, im.tobytes())
    detections = darknet.detect_image(net, meta, darknet_image, thresh=0.25)
    darknet.free_image(darknet_image)
    return detections  # [(b'label', conf, (x,y,w,h)), ...]

def mask_and_occlusion(im, detections):
    masks, tops = [], []
    for label, conf, bbox in detections:
        x,y,w,h = bbox
        # SAM prompt = bbox xywh to x1y1x2y2
        mask = sam_model(im, [[x-w/2, y-h/2, x+w/2, y+h/2]])[0]['mask']
        masks.append(mask.astype(np.uint8))

    # Lógica topo/base: menor nº de blobs == objeto em cima
    blobs = [cv2.connectedComponents(m)[0]-1 for m in masks]
    top_idx = blobs.index(min(blobs))
    for i, m in enumerate(masks):
        color = (0,255,0) if i == top_idx else (0,0,255)
        im[m.astype(bool)] = im[m.astype(bool)]*0.4 + np.array(color)*0.6
    return im

# ---------- main ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="conf/yolov2_surgery.cfg")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", default="data/raw/test")
    ap.add_argument("--outdir", default="runs/infer_darknet")
    args = ap.parse_args()

    net, meta = load_net(args.cfg, args.weights)
    outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    for img_path in pathlib.Path(args.source).glob("*.jpg"):
        im = cv2.imread(str(img_path))
        dets = detect(net, meta, im)
        im_vis = mask_and_occlusion(im, dets)
        cv2.imwrite(str(outdir / img_path.name), im_vis)

if __name__ == "__main__":
    main()
