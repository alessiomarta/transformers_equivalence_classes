#!/bin/bash

cd ..
for dir in experiments_data/test/*; do
    echo $dir
    if [[ "$dir" == *"cifar"* || "$dir" == *"mnist"* ]]; then
        python experiments/vit_exploration.py --experiment-path $dir --out-dir ../res
        sleep 2
        for resdir in ../res/*; do
            if [[ "$resdir" == *"$(basename $dir)"* ]]; then
                echo $resdir
                python experiments/vit_exploration_interpretation.py --experiment-path $dir --results-dir $resdir
            fi
        done
    fi
    
done