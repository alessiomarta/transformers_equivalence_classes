#!/bin/bash

# Define the base paths
experiment_base_path="experiments_data/test"
results_base_path="../res"

# Find the experiment folder name
experiment_folder=$(basename $(find $results_base_path -type d -name "*2025*" | head -n 1))

# Construct the full paths
experiment_path="$experiment_base_path/$(echo $experiment_folder | cut -d'-' -f1-4)"
results_dir="$results_base_path/$experiment_folder"

# Run the python script with the constructed paths
python experiments/vit_exploration_interpretation.py --experiment-path $experiment_path --results-dir $results_dir
for experiment_folder in $(find $results_base_path -type d -name "*2025*"); do
    echo "Processing experiment folder: $(basename $experiment_folder)"
    experiment_path="$experiment_base_path/$(basename $experiment_folder | cut -d'-' -f1-4)"
    results_dir="$results_base_path/$(basename $experiment_folder)"
    python experiments/vit_exploration_interpretation.py --experiment-path $experiment_path --results-dir $results_dir
done