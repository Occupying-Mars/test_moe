#!/bin/bash

# Define experiment names based on file names
declare -A experiments=(
    ["train_muon.py"]="muon_experiment"
    ["train_relu2.py"]="relu2_experiment"
    ["train_rotary.py"]="rotary_experiment"
    ["train_baseline.py"]="baseline_experiment"
)

# Iterate over the experiment scripts and run them sequentially
for script in "${!experiments[@]}"; do
    experiment_name="${experiments[$script]}"
    
    echo "Running training: $script with experiment name: $experiment_name"
    python "$script" --experiment_name "$experiment_name"
    
    echo "Running logger for: $experiment_name"
    python logger.py --experiment_name "$experiment_name"
    
    echo "Finished: $experiment_name"
    echo "---------------------------------"
done

echo "All experiments completed."
