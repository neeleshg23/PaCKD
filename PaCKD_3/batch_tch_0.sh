#!/bin/bash

APPS=("bc-3.txt.xz") 
GPUS=("0" "1" "2" "3" "4" "5")
MODELS=("l" "m" "r")
CLUSTERS=("i") 

gpu_index=3

for cluster in "${CLUSTERS[@]}"; do
    for app1 in "${APPS[@]}"; do
        for model in "${MODELS[@]}"; do
            GPU="${GPUS[$gpu_index]}"
            python src/train_tchs.py $app1 $cluster $model $model $model $GPU
            python src/validate_tchs.py $app1 $cluster $model $model $model $GPU
    	done 
    done
done
