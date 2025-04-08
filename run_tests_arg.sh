#!/bin/bash

# Default values
start_seed=1000
end_seed=5000
ckpt_path=""

# Function to display usage
usage() {
    echo "Usage: $0 -s start_seed -e end_seed -c ckpt_path"
    exit 1
}

# Parse command line arguments
while getopts "s:e:c:" opt; do
    case $opt in
        s) start_seed=$OPTARG ;;
        e) end_seed=$OPTARG ;;
        c) ckpt_path=$OPTARG ;;
        *) usage ;;
    esac
done

# Check if ckpt_path is provided
if [ -z "$ckpt_path" ]; then
    echo "Error: Checkpoint path must be provided."
    usage
fi

# Loop through the seed range
for seed in $(seq $start_seed $end_seed); do
    python KMVE_RG/test_metric.py \
        --model MoEModel \
        --method moe \
        --comment moe \
        --num_experts 3 \
        --num_shared_experts 1 \
        --d_ff 1024 \
        --ckpt "$ckpt_path" \
        --seed $seed
done

