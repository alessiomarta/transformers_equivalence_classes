#!/bin/bash

cd ../experiments

for dir in experiments_data/*; do
    basename_dir=$(basename "$dir")
    #echo "Checking: $basename_dir"

    # Only proceed if it's a winobias experiment
    if [[ "$dir" == *winobias* ]]; then
        echo "Running exploration for: $basename_dir"

        # Run exploration
        python3 bert_exploration.py --experiment-path "$dir" --out-dir ../res --device cuda:0
        sleep 2

        # Find all matching resdirs and sort them by timestamp (descending)
        matched_resdirs=($(ls -d ../res/*"$basename_dir"* 2>/dev/null | sort -t '-' -k6,6 -k7,7 -r))

        if [[ ${#matched_resdirs[@]} -gt 0 ]]; then
            latest_resdir="${matched_resdirs[0]}"
            echo "Latest matching result dir: $latest_resdir"

            # Run interpretation on the most recent result directory
            python3 bert_exploration_interpretation.py --experiment-path "$dir" --results-dir "$latest_resdir" --device cuda:0
        else
            echo "No matching result directory found for $basename_dir"
        fi
    fi
done

for dir in experiments_data/*; do
    basename_dir=$(basename "$dir")
    #echo "Checking: $basename_dir"

    # Only proceed if it's a hatespeech experiment
    if [[ "$dir" == *hatespeech* ]]; then
        echo "Running exploration for: $basename_dir"

        # Run exploration
        python3 bert_exploration.py --experiment-path "$dir" --out-dir ../res --device cuda:0
        sleep 2

        # Find all matching resdirs and sort them by timestamp (descending)
        matched_resdirs=($(ls -d ../res/*"$basename_dir"* 2>/dev/null | sort -t '-' -k6,6 -k7,7 -r))

        if [[ ${#matched_resdirs[@]} -gt 0 ]]; then
            latest_resdir="${matched_resdirs[0]}"
            echo "Latest matching result dir: $latest_resdir"

            # Run interpretation on the most recent result directory
            python3 bert_exploration_interpretation.py --experiment-path "$dir" --results-dir "$latest_resdir" --device cuda:0
        else
            echo "No matching result directory found for $basename_dir"
        fi
    fi
done