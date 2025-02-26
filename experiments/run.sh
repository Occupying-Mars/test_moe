#!/bin/bash

# Define experiment names for all scripts
declare -A experiments=(
    ["_train_muon.py"]="muon_experiment"
    ["_train_relu2.py"]="relu2_experiment"
    ["_train_rot.py"]="rotary_experiment"
    ["_train1.py"]="baseline_experiment"
    ["_train_untied.py"]="untied_experiment"
    ["_train_tanh_cap.py"]="tanhcapping_experiment"
    ["_train_unet.py"]="unet_experiment"
    ["_train_flex.py"]="flex_experiment"
    ["_train_fp8.py"]="fp8_experiment"
    ["_train_qkv_split.py"]="qkv_split_experiment"
    ["_train_sliding_block.py"]="sliding_block_experiment"
    ["_train_sparse_value.py"]="sparse_value_experiment"
    ["_train_unet_value_embed.py"]="unet_value_embed_experiment"
    ["_train_value_embed.py"]="value_embed_experiment"
    ["_train_warmup.py"]="warmup_experiment"
)

# Iterate over the experiment scripts and run them sequentially
for script in "${!experiments[@]}"; do
    experiment_name="${experiments[$script]}"
    
    echo "Running training: $script with experiment name: $experiment_name"
    # python "$script" --experiment_name "$experiment_name"
    
    echo "Running logger for: $experiment_name"
    python logger.py --experiment_name "$experiment_name"
    
    echo "Finished: $experiment_name"
    echo "---------------------------------"
done

echo "All experiments completed."
