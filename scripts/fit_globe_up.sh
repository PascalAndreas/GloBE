#!/bin/bash

# GloBE Up Projection Fitting Script
# This script fits global basis banks for Up projections using weight-only reconstruction

set -e  # Exit on any error

# Default configuration
MODEL_NAME="Qwen1.5-MoE-A2.7B"
CONFIG_NAME="default"
OUTPUT_DIR="outputs/globe_up_fitting"
NUM_GPUS=1
WANDB_PROJECT="globe"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --config)
            CONFIG_NAME="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --wandb-project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --model-name MODEL     Model name (default: $MODEL_NAME)"
            echo "  --config CONFIG        Config name (default: $CONFIG_NAME)"
            echo "  --output-dir DIR       Output directory (default: $OUTPUT_DIR)"
            echo "  --num-gpus N           Number of GPUs (default: $NUM_GPUS)"
            echo "  --wandb-project PROJ   Wandb project name (default: $WANDB_PROJECT)"
            echo "  --help                 Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Print configuration
echo "GloBE Up Projection Fitting"
echo "=========================="
echo "Model: $MODEL_NAME"
echo "Config: $CONFIG_NAME"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $NUM_GPUS"
echo "Wandb Project: $WANDB_PROJECT"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Set environment variables
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
export WANDB_PROJECT="$WANDB_PROJECT"
export WANDB_RUN_NAME="globe_up_${MODEL_NAME}_$(date +%Y%m%d_%H%M%S)"

# Run training command
if [ "$NUM_GPUS" -gt 1 ]; then
    # Multi-GPU training
    python -m torch.distributed.launch \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29500 \
        -m globe.train.fit_banks \
        --config-name=$CONFIG_NAME \
        model.name="$MODEL_NAME" \
        hydra.run.dir="$OUTPUT_DIR" \
        train.family="up" \
        wandb.run_name="$WANDB_RUN_NAME" \
        +distributed=true
else
    # Single GPU training
    python -m globe.train.fit_banks \
        --config-name=$CONFIG_NAME \
        model.name="$MODEL_NAME" \
        hydra.run.dir="$OUTPUT_DIR" \
        train.family="up" \
        wandb.run_name="$WANDB_RUN_NAME"
fi

echo ""
echo "Up projection fitting completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "Check wandb for training logs: https://wandb.ai/$WANDB_PROJECT"
