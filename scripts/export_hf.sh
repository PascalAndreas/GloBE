#!/bin/bash

# GloBE HuggingFace Export Script
# This script exports a trained GloBE model to HuggingFace format with precomposed weights

set -e  # Exit on any error

# Default configuration
MODEL_NAME="Qwen1.5-MoE-A2.7B"
GLOBE_WEIGHTS_PATH=""
OUTPUT_DIR="outputs/exported_models"
EXPORT_DTYPE="bf16"
VALIDATE_EXPORT="true"
PUSH_TO_HUB="false"
HUB_REPO_ID=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --globe-weights)
            GLOBE_WEIGHTS_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --export-dtype)
            EXPORT_DTYPE="$2"
            shift 2
            ;;
        --no-validate)
            VALIDATE_EXPORT="false"
            shift
            ;;
        --push-to-hub)
            PUSH_TO_HUB="true"
            shift
            ;;
        --hub-repo-id)
            HUB_REPO_ID="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 --globe-weights PATH [options]"
            echo "Required:"
            echo "  --globe-weights PATH   Path to trained GloBE weights"
            echo "Options:"
            echo "  --model-name MODEL     Base model name (default: $MODEL_NAME)"
            echo "  --output-dir DIR       Output directory (default: $OUTPUT_DIR)"
            echo "  --export-dtype DTYPE   Export dtype (default: $EXPORT_DTYPE)"
            echo "  --no-validate          Skip export validation"
            echo "  --push-to-hub          Push to HuggingFace Hub"
            echo "  --hub-repo-id REPO     Hub repository ID (required with --push-to-hub)"
            echo "  --help                 Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$GLOBE_WEIGHTS_PATH" ]; then
    echo "Error: --globe-weights PATH is required"
    exit 1
fi

if [ ! -f "$GLOBE_WEIGHTS_PATH" ]; then
    echo "Error: GloBE weights file not found: $GLOBE_WEIGHTS_PATH"
    exit 1
fi

if [ "$PUSH_TO_HUB" = "true" ] && [ -z "$HUB_REPO_ID" ]; then
    echo "Error: --hub-repo-id is required when using --push-to-hub"
    exit 1
fi

# Print configuration
echo "GloBE HuggingFace Export"
echo "======================="
echo "Model: $MODEL_NAME"
echo "GloBE Weights: $GLOBE_WEIGHTS_PATH"
echo "Output: $OUTPUT_DIR"
echo "Export dtype: $EXPORT_DTYPE"
echo "Validate: $VALIDATE_EXPORT"
echo "Push to Hub: $PUSH_TO_HUB"
if [ "$PUSH_TO_HUB" = "true" ]; then
    echo "Hub Repo: $HUB_REPO_ID"
fi
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build export command
EXPORT_CMD="python -m globe.infer.export_hf"
EXPORT_CMD="$EXPORT_CMD --model-name '$MODEL_NAME'"
EXPORT_CMD="$EXPORT_CMD --globe-weights '$GLOBE_WEIGHTS_PATH'"
EXPORT_CMD="$EXPORT_CMD --output-dir '$OUTPUT_DIR'"
EXPORT_CMD="$EXPORT_CMD --export-dtype '$EXPORT_DTYPE'"

if [ "$VALIDATE_EXPORT" = "false" ]; then
    EXPORT_CMD="$EXPORT_CMD --no-validate"
fi

if [ "$PUSH_TO_HUB" = "true" ]; then
    EXPORT_CMD="$EXPORT_CMD --push-to-hub --hub-repo-id '$HUB_REPO_ID'"
fi

# Run export
echo "Running export command..."
eval $EXPORT_CMD

echo ""
echo "Export completed successfully!"
echo "Exported model saved to: $OUTPUT_DIR"

if [ "$PUSH_TO_HUB" = "true" ]; then
    echo "Model pushed to HuggingFace Hub: $HUB_REPO_ID"
fi

# Display export summary
if [ -f "$OUTPUT_DIR/globe_export_info.json" ]; then
    echo ""
    echo "Export Summary:"
    echo "==============="
    python -c "
import json
with open('$OUTPUT_DIR/globe_export_info.json', 'r') as f:
    info = json.load(f)
print(f\"Precomposed experts: {info.get('precomposed_experts', 'N/A')}\")
print(f\"Total experts: {info.get('total_experts', 'N/A')}\")
if 'validation' in info:
    validation = info['validation']
    print(f\"Validation passed: {validation.get('passed', 'N/A')}\")
"
fi
