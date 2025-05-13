#!/bin/bash

cd ../experiments

for dir in experiments_data/*; do
    basename_dir=$(basename "$dir")
    #echo "Checking: $basename_dir"

    # Only proceed if it's a CIFAR experiment 
    if [[ "$dir" == *hatespeech* && "$dir" != *1p0-15-all* ]]; then
        # Find all matching resdirs and sort them by timestamp (descending)
        matched_resdirs=($(ls -d ../res/*"$basename_dir"* 2>/dev/null | sort -t '-' -k6,6 -k7,7 -r))

        if [[ ${#matched_resdirs[@]} -gt 0 ]]; then
            latest_resdir="${matched_resdirs[0]}"
            echo "Latest matching result dir: $latest_resdir"
            # Remove all interpretation/ dirs in subdirectories of the latest result directory
            find "$latest_resdir" -type d -name "interpretation" -exec rm -rf {} +
            # Run interpretation on the most recent result directory
            python3 bert_exploration_interpretation.py --experiment-path "$dir" --results-dir "$latest_resdir" --device cpu
        else
            echo "No matching result directory found for $basename_dir"
        fi
    fi
done