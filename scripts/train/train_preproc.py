#!/usr/bin/env python
"""
Gera imagens de fundo removido usando as máscaras COCO
e grava em data/preprocessed/images/<split>/.
Depois você pode treinar outro YOLO (detecção) sobre essas
imagens “clean” para testar o pré-processamento SAM.
"""

import json, cv2, numpy as np
from pathlib import Path
from tqdm import tqdm
from pycocotools import mask as cocomask

DATA_DIR   = Path("data/processed")
OUT_DIR    = Path("data/preprocessed/images")
SPLITS     = ["train", "val", "test"]

def coco_to_binary(mask_ann):
    """COCO RLE ou polygon → máscara binária"""
    if isinstance(mask_ann, list):               # polygon
        rles = cocomask.frPyObjects(mask_ann, mask_ann["height"], mask_ann["width"])
        rle  = cocomask.merge(rles)
    elif isinstance(mask_ann["counts"], list):   # uncompressed RLE
        rle = cocomask.frPyObjects(mask_ann, mask_ann["height"], mask_ann["width"])
    else:
        rle = mask_ann                           # já RLE
    return cocomask.decode(rle)

def process_split(split):
    OUT_DIR.joinpath(split).mkdir(parents=True, exist_ok=True)
    coco = json.load(open(DATA_DIR / "masks_coco" / f"{split}.json"))
    img_id_to_name = {img["id"]: img["file_name"] for img in coco["images"]}

    for ann in tqdm(coco["annotations"], desc=f"Split {split}"):
        img_path  = DATA_DIR / "images" / split / img_id_to_name[ann["image_id"]]
        mask      = coco_to_binary(ann["segmentation"])
        img       = cv2.imread(str(img_path))
        img[mask == 0] = (0, 0, 0)               # zera fundo
        cv2.imwrite(str(OUT_DIR / split / img_path.name), img)

def main():
    for s in SPLITS:
        process_split(s)

if __name__ == "__main__":
    main()
