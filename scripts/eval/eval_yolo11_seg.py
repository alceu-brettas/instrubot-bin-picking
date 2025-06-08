# scripts/eval/eval_yolo11_bbox.py

from ultralytics import YOLO

MODEL_PATH = "experiments/C_yolo11_seg/run/weights/best.pt"
DATA_YAML = "conf/data_mask.yaml"
SAVE_PROJECT = "experiments/C_yolo11_seg/eval"
SAVE_NAME = "eval"

def main():
    model = YOLO(MODEL_PATH)

    metrics = model.val(
            data=DATA_YAML,
            split="val",          # ou "val"
            device="cpu",
            project=SAVE_PROJECT,
            name=SAVE_NAME,
            exist_ok=False
        )

    print("\n✅ Avaliação finalizada.")
    print(f"📂 Resultados salvos em: {metrics.save_dir}\n")

    # ✅ Resultados gerais
    mp, mr, map50, map95 = metrics.box.mean_results()
    print(f"🎯 Precision (mean): {mp:.3f}")
    print(f"📦 Recall (mean): {mr:.3f}")
    print(f"📊 mAP@0.5: {map50:.3f}")
    print(f"📊 mAP@0.5:0.95: {map95:.3f}")

    # ✅ Resultados por classe
    print("\n📈 Resultados por classe:\n")
    print(f"{'Classe':<15} {'Prec.':>7} {'Recall':>7} {'AP@0.5':>9} {'AP@0.5:0.95':>13}")
    print("-" * 52)

    for i in range(model.nc):
        name = model.names.get(i, f'class_{i}')
        try:
            p_i, r_i, ap50_i, ap95_i = metrics.box.class_result(i)
            print(f"{name:<15} {p_i:7.3f} {r_i:7.3f} {ap50_i:9.3f} {ap95_i:13.3f}")
        except Exception:
            print(f"{name:<15} {'—':>7} {'—':>7} {'—':>9} {'—':>13}")

if __name__ == "__main__":
    main()
