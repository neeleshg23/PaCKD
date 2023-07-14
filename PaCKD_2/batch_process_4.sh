#!/bin/bash
APPS=("sssp-3.txt.xz")
GPUS=("0" "1" "4" "5")
CLUSTERS=("a" "d" "i")

gpu_index=0
for app1 in "${APPS[@]}"; do
    for cluster in "${CLUSTERS[@]}"; do
        GPU="${GPUS[$gpu_index]}"
        python src/preprocess.py $app1 $cluster $GPU
    done
done
