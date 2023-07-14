#!/bin/bash

APPS=("pr-3.txt.xz") 
GPUS=("0" "1" "4" "5")
MODELS=("l" "m" "r")
CLUSTERS=("a") 

gpu_index=0
alpha=("0.1" "0.25" "0.5" "0.75" "0.9")
for cluster in "${CLUSTERS[@]}"; do
    for app1 in "${APPS[@]}";do 
	    for model in "${MODELS[@]}"; do
	        for a in "${alpha[@]}"; do
                GPU="${GPUS[$gpu_index]}"
                python src/train_stu.py $app1 $cluster $alpha $model $model $model $GPU
            done
        done
    done
done
