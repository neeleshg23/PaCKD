#!/bin/bash

TraceDir="data"

GPUS=("0" "1" "4" "5")
MODELS=("l" "m" "r")
CLUSTERS=("a" "d" "i")

gpu_index=0
alpha=0.5

for cluster in "${CLUSTERS[@]}"; do
    for app1 in `ls $TraceDir`; do
        for model in "${MODELS[@]}"; do
            GPU="${GPUS[$gpu_index]}"
            python3 src/train_stu.py $app1 $cluster $alpha $model $model $model $GPU
        done
    done
done