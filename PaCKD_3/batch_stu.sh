#!/bin/bash

APPS=("pr-3.txt.xz" "sssp-3.txt.xz") 
GPUS=("0" "1" "2" "3" "4" "5")
MODELS=("l" "m" "r")
CLUSTERS=("a" "d" "i") 
ALPHAS=("0.1" "0.3" "0.7" "0.9")

gpu_index=2

for cluster in "${CLUSTERS[@]}"; do
    for app1 in "${APPS[@]}"; do
        for model in "${MODELS[@]}"; do
            for alpha in "${ALPHAS[@]}"; do 
                GPU="${GPUS[$gpu_index]}"
                python src/train_stu.py $app1 $cluster $alpha $alpha $alpha $model $model $model $model $GPU
            done 
        done
    done
done
