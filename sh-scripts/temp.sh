#!/bin/bash

cd ../experiments

# Helper function to find the most recent result directory
get_latest_resdir() {
  local experiment_path=$1
  local exp_name=$(basename "$experiment_path")
  local res_prefix="../res/${exp_name}"

  # Find the latest matching result directory
  resdir=$(ls -td ${res_prefix}* | head -n1)
  echo "$resdir"
}

# First experiment
exp1="experiments_data/mnist-1p0-32-one"
python3 vit_exploration.py --experiment-path $exp1 --out-dir ../res
resdir=$(get_latest_resdir $exp1)
python3 vit_exploration_interpretation.py --experiment-path $exp1 --results-dir $resdir

# Second experiment
exp2="experiments_data/mnist-1p0-32-q2"
python3 vit_exploration.py --experiment-path $exp2 --out-dir ../res
resdir=$(get_latest_resdir $exp2)
python3 vit_exploration_interpretation.py --experiment-path $exp2 --results-dir $resdir