#!/usr/bin/env python3

from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
COCO_JSON   = Path("experiments/C_yolo11_seg/infer/segmentations_test.json")
OUTPUT_JSON = Path("experiments/C_yolo11_seg/occlusion/occlusion_results.json")
IMAGES_ROOT = Path("data/processed_seg/images/test")
OUT_DIR     = Path("experiments/C_yolo11_seg/occlusion/top_objects_only")
# ---------------------------------------------------------------------------

# Kernels morfológicos
_DIL_KERNEL  = np.ones((5, 5), np.uint8)   # fecha gaps de até 2 px
_EROS_KERNEL = np.ones((3, 3), np.uint8)   # quebra filetes de 1 px

# ---------------------------------------------------------------------------
# Utilidades de máscara e grafo
# ---------------------------------------------------------------------------
def polygon_to_mask_raw(poly: List[float], h: int, w: int) -> np.ndarray:
    """Polígono flatten → máscara uint8 (0/1) sem pós-processo."""
    pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(m, [pts], 1)
    return m

def connected_components_count(mask_bool: np.ndarray) -> int:
    n, _ = cv2.connectedComponents(mask_bool.astype(np.uint8))
    return n - 1

def topo_sort(graph: Dict[int, set]) -> List[int]:
    indeg = {u: 0 for u in graph}
    for u, nbrs in graph.items():
        for v in nbrs:
            indeg[v] += 1
    q = deque([u for u, d in indeg.items() if d == 0])
    order: List[int] = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in graph[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return order  # bottom→top

def decide_occlusion(mask_dil, mask_ori, areas, ccs):
    """Retorna (grafo bottom→top, lista top→bottom)."""
    n = len(mask_ori)
    below = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if not np.logical_and(mask_dil[i], mask_dil[j]).any():
                continue
            # Regra 0 – multi-componentes nunca é topo
            if ccs[i] > 1 and ccs[j] == 1:
                below[i].add(j);  continue
            if ccs[j] > 1 and ccs[i] == 1:
                below[j].add(i);  continue
            # Regra 1 – fração visível
            inter = np.logical_and(mask_ori[i], mask_ori[j]).sum()
            r_i = inter / areas[i]
            r_j = inter / areas[j]
            if r_i < r_j:
                below[i].add(j)
            elif r_j < r_i:
                below[j].add(i)
            else:                      # empate → mais partes fica embaixo
                if ccs[i] > ccs[j]:
                    below[i].add(j)
                elif ccs[j] > ccs[i]:
                    below[j].add(i)
    order_tb = list(reversed(topo_sort(below)))
    return below, order_tb

# ---------------------------------------------------------------------------
# COCO helpers
# ---------------------------------------------------------------------------
def load_coco(path: Path):
    with open(path, "r") as f:
        return json.load(f)

def build_index(coco: dict):
    anns_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)
    images_dict = {img["id"]: img for img in coco["images"]}
    anns_by_id  = {ann["id"]: ann for ann in coco["annotations"]}
    return anns_by_img, images_dict, anns_by_id

# ---------------------------------------------------------------------------
# Análise principal
# ---------------------------------------------------------------------------
def analyse_occlusions(anns_by_img, images_dict):
    results = []
    for image_id, anns in tqdm(anns_by_img.items(), desc="Analisando oclusões"):
        h, w = images_dict[image_id]["height"], images_dict[image_id]["width"]
        mask_ori, mask_dil, areas, ccs, ann_ids = [], [], [], [], []
        for ann in anns:
            m_raw  = polygon_to_mask_raw(ann["segmentation"][0], h, w)
            m_ori  = cv2.erode(m_raw, _EROS_KERNEL, 1).astype(bool)
            m_dil  = cv2.dilate(m_raw, _DIL_KERNEL, 1).astype(bool)
            mask_ori.append(m_ori); mask_dil.append(m_dil)
            areas.append(int(m_ori.sum()))
            ccs.append(connected_components_count(m_ori))
            ann_ids.append(ann["id"])

        below, order_tb = decide_occlusion(mask_dil, mask_ori, areas, ccs)
        relations  = [{"top": int(ann_ids[j]), "bottom": int(ann_ids[i])}
                      for i, tops in below.items() for j in tops]
        components = {int(ann_ids[k]): int(ccs[k]) for k in range(len(ann_ids))}

        results.append({
            "image_id": int(image_id),
            "file_name": images_dict[image_id]["file_name"],
            "order_top_to_bottom": [int(ann_ids[k]) for k in order_tb],
            "relations": relations,
            "components": components,
        })
    return results

# ---------------------------------------------------------------------------
# Overlay + export
# ---------------------------------------------------------------------------
def draw_top_object(src: Path, ann_top: dict, h: int, w: int, dst: Path):
    img = cv2.imread(str(src))
    if img is None:
        print(f"⚠️  Falha ao ler {src}"); return
    m_raw = polygon_to_mask_raw(ann_top["segmentation"][0], h, w)
    m_ori = cv2.erode(m_raw, _EROS_KERNEL, 1).astype(bool)
    overlay = img.copy();  overlay[m_ori] = (0, 255, 0)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
    cnts, _ = cv2.findContours(m_ori.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        x, y, w_box, _ = cv2.boundingRect(cnts[0])
        cv2.putText(img, "TOP OBJECT", (x + w_box + 10, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), img)

def export_outputs(occl_results, anns_by_id, images_dict):
    # JSON
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(occl_results, f, indent=2)
    print(f"✅ JSON salvo em: {OUTPUT_JSON}")

    # Imagens
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_exp = 0
    for rec in occl_results:
        if not rec["relations"]:
            continue
        top_id = rec["order_top_to_bottom"][0] if rec["order_top_to_bottom"] else None
        if top_id is None:
            continue
        ann_top  = anns_by_id[top_id]
        img_info = images_dict[rec["image_id"]]
        draw_top_object(IMAGES_ROOT / img_info["file_name"], ann_top,
                        img_info["height"], img_info["width"],
                        OUT_DIR / img_info["file_name"])
        n_exp += 1
    print(f"🖼️  {n_exp} imagens exportadas em '{OUT_DIR}'.")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    coco = load_coco(COCO_JSON)
    anns_by_img, images_dict, anns_by_id = build_index(coco)
    occl_results = analyse_occlusions(anns_by_img, images_dict)
    export_outputs(occl_results, anns_by_id, images_dict)

if __name__ == "__main__":
    main()
