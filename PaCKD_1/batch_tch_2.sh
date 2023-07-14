#!/bin/bash

TraceDir="data"

APPS=("bc-3.txt.xz")
GPUS=("0" "1" "4" "5")
MODELS=("l" "m" "r")

gpu_index=2

for app1 in "${APPS[@]}"; do
    for model in "${MODELS[@]}"; do
        GPU="${GPUS[$gpu_index]}"
        python src/train_tchs.py $app1 $model $GPU
        python src/validate_tchs.py $app1 $model $GPU
    done 
done

python src/train_tchs.py sssp-3.txt.xz r 4
python src/validate_tchs.py sssp-3.txt.xz r 4
