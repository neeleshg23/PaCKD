#!/bin/bash

APPS=("pr-3.txt.xz") 
GPUS=("0" "1" "4" "5")
MODELS=("l" "m")
CLUSTERS=("d") 

gpu_index=0

for cluster in "${CLUSTERS[@]}"; do
    for app1 in "${APPS[@]}"; do
        for model in "${MODELS[@]}"; do
            GPU="${GPUS[$gpu_index]}"
            python src/train_tchs.py $app1 $cluster $model $model $GPU
            python src/validate_tchs.py $app1 $cluster $model $model $GPU
    	done 
    done
done
