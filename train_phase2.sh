#!/bin/bash

###############################################################################
# Phase 2 Training Wrapper with Timestamped Logging
###############################################################################

set -e  # Exit on error

# Generate timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/${TIMESTAMP}"

# Create log directory
mkdir -p "$LOG_DIR"

# Configuration
CONFIG_FILE="src/axolotl_configs/rakuten_7b_phase2.yml"
LOG_FILE="$LOG_DIR/training.log"

echo "================================================================================"
echo "Phase 2 Training with Logging"
echo "================================================================================"
echo "Start time:  $(date '+%Y-%m-%d %H:%M:%S')"
echo "Config:      $CONFIG_FILE"
echo "Log dir:     $LOG_DIR"
echo "Log file:    $LOG_FILE"
echo "================================================================================"
echo ""

# Run training and log to both console and file
axolotl train "$CONFIG_FILE" 2>&1 | tee "$LOG_FILE"

TRAIN_EXIT_CODE=$?

echo ""
echo "================================================================================"
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully"
else
    echo "✗ Training failed with exit code: $TRAIN_EXIT_CODE"
fi
echo "================================================================================"
echo "End time:    $(date '+%Y-%m-%d %H:%M:%S')"
echo "Log saved:   $LOG_FILE"
echo "================================================================================"
echo ""

# Display log path for easy reference
echo "To view training log:"
echo "  tail -f $LOG_FILE"
echo ""
echo "To view all training sessions:"
echo "  ls -lh logs/"
echo ""

exit $TRAIN_EXIT_CODE
