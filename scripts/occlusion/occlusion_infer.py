
import os
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from skimage.measure import label, regionprops

# Caminhos
INFER_DIR = Path("instrument_detect/experiments/C_yolo11_seg/infer")
OUT_PATH = INFER_DIR / "occlusion_results.json"

# Carga das áreas médias esperadas por classe (exemplo fictício; ajustar com base real)
AREA_MEDIA = {
    0: 3200,  # Bisturi nº4
    1: 4000,  # Pinça de Dissecção Reta
    2: 5000,  # Tesoura Mayo Reta
    3: 5000,  # Tesoura Mayo Curva
}

# Limiar de área para considerar objeto como ocluído
LIMIAR_AREA = 0.4

def processar_mascara(mascara):
    # Conta blobs conectados
    blobs = label(mascara)
    num_blobs = np.max(blobs)
    area_visivel = np.sum(mascara > 0)
    return num_blobs, area_visivel

def inferir_oclusao(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    resultados = []

    for obj in data:
        cls = obj["cls"]
        mask_path = obj.get("mask_path")
        if not mask_path or not os.path.exists(mask_path):
            continue

        # Carrega a máscara binária
        mascara = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mascara is None:
            continue

        mascara_bin = (mascara > 127).astype(np.uint8)
        num_blobs, area_visivel = processar_mascara(mascara_bin)

        area_media = AREA_MEDIA.get(cls, 1)  # Evita divisão por zero
        proporcao = area_visivel / area_media

        if num_blobs > 1 or proporcao < LIMIAR_AREA:
            status = "BOTTOM"
        else:
            status = "TOP"

        resultados.append({
            "arquivo": json_path.name,
            "classe": cls,
            "area_visivel": int(area_visivel),
            "area_media": int(area_media),
            "proporcao": round(proporcao, 2),
            "blobs": num_blobs,
            "oclusao": status
        })

    return resultados

def main():
    jsons = sorted(INFER_DIR.glob("*.json"))
    todos_resultados = []

    for json_file in tqdm(jsons, desc="Processando inferências"):
        resultados = inferir_oclusao(json_file)
        todos_resultados.extend(resultados)

    with open(OUT_PATH, 'w') as f:
        json.dump(todos_resultados, f, indent=2)

    print(f"Resultados salvos em: {OUT_PATH}")

if __name__ == "__main__":
    main()
