#!/bin/bash
# Axolotl training wrapper script
# Usage: ./scripts/run_axolotl_train.sh [CONFIG_FILE] [NUM_GPUS]

set -e

# Configuration
CONFIG_FILE="${1:-src/axolotl_configs/qwen_finetune.yml}"
NUM_GPUS="${2:-4}"

# Prefer repo-local venv binaries (works even if not "activated").
if [ -d ".venv/bin" ]; then
    export PATH="$(pwd)/.venv/bin:$PATH"
fi

# Basic validation
if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]]; then
    echo "Error: NUM_GPUS must be a positive integer, got: $NUM_GPUS"
    exit 1
fi
if [ "$NUM_GPUS" -lt 1 ]; then
    echo "Error: NUM_GPUS must be >= 1, got: $NUM_GPUS"
    exit 1
fi

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
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
fi
echo ""

# Ensure src/ is in Python path for custom callbacks
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
echo "PYTHONPATH configured: $(pwd)"
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
echo "[Validation] Checking required files..."

# Check evaluation data (used by GenerationEvalCallback)
if [ ! -f "data/train/hawks_val.json" ]; then
    echo "⚠️  Warning: data/train/hawks_val.json not found"
    echo "   GenerationEvalCallback will skip evaluation (non-fatal)"
fi

# Check DeepSpeed config (required)
if [ ! -f "src/deepspeed_configs/zero2.json" ]; then
    echo "❌ Error: DeepSpeed config not found"
    exit 1
fi

echo "✅ Validation complete"
echo ""

echo "[2/2] Starting training..."
echo "==============================================="

# Explicitly request multi-GPU when NUM_GPUS > 1 to avoid relying on a user-level
# accelerate config.
ACCELERATE_LAUNCH_ARGS=(--num_processes="$NUM_GPUS" --mixed_precision=bf16)
if [ "$NUM_GPUS" -gt 1 ]; then
    ACCELERATE_LAUNCH_ARGS=(--multi_gpu "${ACCELERATE_LAUNCH_ARGS[@]}")
fi

# Run training with accelerate
accelerate launch \
    "${ACCELERATE_LAUNCH_ARGS[@]}" \
    -m axolotl.cli.train \
    "$CONFIG_FILE"

echo ""
echo "==============================================="
echo "✅ Training completed!"
echo "Output directory: ./outputs/axolotl_qwen_finetune"
echo "==============================================="
