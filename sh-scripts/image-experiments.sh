#!/bin/bash

cd ../experiments
for dir in experiments_data/*; do
    echo $dir
    if [[ "$dir" == *"cifar"* || "$dir" == *"mnist"* ]]; then
        python3 vit_exploration.py --experiment-path $dir --out-dir ../res
        sleep 2
        for resdir in ../res/*; do
            if [[ "$resdir" == *"$(basename $dir)"* ]]; then
                echo $resdir
                python3 vit_exploration_interpretation.py --experiment-path $dir --results-dir $resdir
            fi
        done
    fi
    
done