#!/bin/bash

APPS=("cc-13.txt.xz") 
GPUS=("0" "1" "4" "5")
MODELS=("l" "m" "r")

gpu_index=3

for app1 in "${APPS[@]}"; do
    for model in "${MODELS[@]}"; do
        GPU="${GPUS[$gpu_index]}"
        python src/train_val_stu.py $app1 $model $GPU
    done
done
