#!/bin/bash

# Arrays for datasets and configurations
datasets=("cifar" "mnist")
configs=("all" "q2" "one")
device="cuda:0"

# Base paths
experiment_base="/home/serusr01/transformers_equivalence_classes/experiments/experiments_data"
mkdir -p "./results"

# Loop through all combinations
for dataset in "${datasets[@]}"; do
    for config in "${configs[@]}"; do
        # Find the experiment directory (using 0p1 as the default value)
        experiment_dir="${experiment_base}/${dataset}-0p1-16-${config}"
        
        if [ -d "$experiment_dir" ]; then
            echo "Running vit_perturbation.py for ${dataset}-${config}"
            echo "Experiment directory: $experiment_dir"
            
            # Run the script
            python3 vit_perturbation.py \
                --experiment-path "$experiment_dir" \
                --out-dir "./results" \
                --iterations 1000 \
                --batch-size 4 \
                --device "$device"
            
            echo "Completed ${dataset}-${config}"
            echo "----------------------------------------"
        else
            echo "Warning: Experiment directory not found: $experiment_dir"
        fi
    done
done

# Arrays for datasets and configurations
datasets=("hatespeech" "winobias")
configs=("all" "q2" "one" "target-word")

# Loop through all combinations
for dataset in "${datasets[@]}"; do
    for config in "${configs[@]}"; do
        # Find the experiment directory (using 0p1 as the default value)
        experiment_dir="${experiment_base}/${dataset}-0p1-16-${config}"
        
        if [ -d "$experiment_dir" ]; then
            echo "Running bert_perturbation.py for ${dataset}-${config}"
            echo "Experiment directory: $experiment_dir"
            
            # Run the script
            python3 bert_perturbation.py \
                --experiment-path "$experiment_dir" \
                --out-dir "./results" \
                --iterations 1000 \
                --batch-size 2 \
                --device "$device"
            
            echo "Completed ${dataset}-${config}"
            echo "----------------------------------------"
        else
            echo "Warning: Experiment directory not found: $experiment_dir"
        fi
    done
done

echo "All bert_perturbation.py runs completed!"