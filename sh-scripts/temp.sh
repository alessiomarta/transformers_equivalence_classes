#!/bin/bash

cd ../experiments

for dir in experiments_data/*; do
    basename_dir=$(basename "$dir")
    echo "Checking: $basename_dir"

    # Only proceed if it's a MNIST or CIFAR experiment that also contains "16" in the name
    if [[ ("$dir" == *mnist* || "$dir" == *cifar*) && "$basename_dir" == *16* ]]; then
        echo "Running exploration for: $basename_dir"

        # Run exploration
        python3 vit_exploration.py --experiment-path "$dir" --out-dir ../res
        sleep 2

        # Find all matching resdirs and sort them by timestamp (descending)
        matched_resdirs=($(ls -d ../res/*"$basename_dir"* 2>/dev/null | sort -t '-' -k6,6 -k7,7 -r))

        if [[ ${#matched_resdirs[@]} -gt 0 ]]; then
            latest_resdir="${matched_resdirs[0]}"
            echo "Latest matching result dir: $latest_resdir"

            # Run interpretation on the most recent result directory
            python3 vit_exploration_interpretation.py --experiment-path "$dir" --results-dir "$latest_resdir"
        else
            echo "No matching result directory found for $basename_dir"
        fi
    else
        echo "Skipping: $basename_dir"
    fi
done