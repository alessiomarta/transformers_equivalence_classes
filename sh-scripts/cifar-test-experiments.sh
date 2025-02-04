#!/bin/bash

cd ..
for dir in experiments_data/test/*; do
    echo $dir
    if [[ "$dir" == *"cifar"* ]]; then
        python experiments/vit_exploration.py --experiment-path $dir --out-dir ../res
    
    fi
    
done
