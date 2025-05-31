#!/usr/bin/env bash
# Treino baseline Lavado em Darknet

# Caminhos relativos ao repo
DATA=conf/data_bbox_yolov2.data       # train/val/test + classes
CFG=conf/yolov2_surgery.cfg           # baseada na cfg de Lavado
WEIGHTS=darknet/weights/darknet19_448.conv.23

# Saída
EXP_DIR=experiments/A_yolov2_baseline
mkdir -p "$EXP_DIR"

# Ative seu env (opcional, se compilou darknet com CUDA do sistema)
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate binpick-yolo2

# Treino (altera paths se o executável estiver em outro lugar)
./darknet/darknet detector train "$DATA" "$CFG" "$WEIGHTS" \
        -dont_show \
        -map \
        -gpus 0 \
        -project "$EXP_DIR"
