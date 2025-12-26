#!/bin/bash
# Axolotl training wrapper script
# Usage: ./scripts/run_axolotl_train.sh [CONFIG_FILE] [NUM_GPUS]

set -e

# Configuration
CONFIG_FILE="${1:-src/axolotl_configs/qwen_finetune.yml}"
NUM_GPUS="${2:-4}"

# Verify config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "==============================================="
echo "Axolotl Training Script"
echo "==============================================="
echo "Config file: $CONFIG_FILE"
echo "Number of GPUs: $NUM_GPUS"
echo ""

# Data preparation
echo "[1/2] Preparing curriculum data..."
if [ ! -f "data/train/hawks_train_curriculum.json" ]; then
    echo "  Generating hawks_train_curriculum.json..."
    python scripts/prepare_curriculum_data.py
else
    echo "  hawks_train_curriculum.json already exists (skip generation)"
fi

echo ""
echo "[2/2] Starting training..."
echo "==============================================="

# Run training with accelerate
accelerate launch \
    --num_processes=$NUM_GPUS \
    --mixed_precision=bf16 \
    -m axolotl.cli.train \
    "$CONFIG_FILE"

echo ""
echo "==============================================="
echo "✅ Training completed!"
echo "Output directory: ./outputs/axolotl_qwen_finetune"
echo "==============================================="
