#!/bin/bash

cd ..
for dir in experiments_data/test/*mnist*; do
    echo $dir
    python experiments/vit_exploration.py --experiment-path $dir --out-dir ../res
done
