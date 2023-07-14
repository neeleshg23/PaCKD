#!/bin/bash

APPS=("pr-3.txt.xz") 
GPUS=("0" "1" "4" "5")
MODELS=("l" "m" "r")

gpu_index=0

for app1 in "${APPS[@]}";do 
	for model in "${MODELS[@]}"; do
	    GPU="${GPUS[$gpu_index]}"
        python src/train_val_stu.py $app1 $model $GPU
    done
done
