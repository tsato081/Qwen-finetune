#!/bin/bash

###############################################################################
# Phase 1 + Phase 2 + Model Merge + HF Upload Pipeline
###############################################################################

set -e  # Exit on error

# Generate timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/${TIMESTAMP}"

# Create log directory
mkdir -p "$LOG_DIR"

echo "================================================================================"
echo "LoRA Finetuning Pipeline (Phase 1 + Phase 2 + Upload)"
echo "================================================================================"
echo "Start time:  $(date '+%Y-%m-%d %H:%M:%S')"
echo "Log dir:     $LOG_DIR"
echo "================================================================================"
echo ""

# ============================================================
# Phase 1: Negative Example Calibration
# ============================================================

echo "Phase 1: Negative Example Calibration"
echo "================================================================================"

PHASE1_CONFIG="src/axolotl_configs/rakuten_7b_phase1.yml"
PHASE1_LOG="$LOG_DIR/phase1.log"

echo "Config:      $PHASE1_CONFIG"
echo "Log file:    $PHASE1_LOG"
echo "================================================================================"
echo ""

axolotl train "$PHASE1_CONFIG" 2>&1 | tee "$PHASE1_LOG"
PHASE1_EXIT_CODE=$?

if [ $PHASE1_EXIT_CODE -eq 0 ]; then
    echo "✓ Phase 1 Training completed successfully"
else
    echo "✗ Phase 1 Training failed with exit code: $PHASE1_EXIT_CODE"
    exit $PHASE1_EXIT_CODE
fi

echo ""

# ============================================================
# Phase 2: Positive Example Learning + Negative Replay
# ============================================================

echo "Phase 2: Positive Example Learning + Negative Replay"
echo "================================================================================"

PHASE2_CONFIG="src/axolotl_configs/rakuten_7b_phase2.yml"
PHASE2_LOG="$LOG_DIR/phase2.log"

echo "Config:      $PHASE2_CONFIG"
echo "Log file:    $PHASE2_LOG"
echo "================================================================================"
echo ""

axolotl train "$PHASE2_CONFIG" 2>&1 | tee "$PHASE2_LOG"
PHASE2_EXIT_CODE=$?

if [ $PHASE2_EXIT_CODE -eq 0 ]; then
    echo "✓ Phase 2 Training completed successfully"
else
    echo "✗ Phase 2 Training failed with exit code: $PHASE2_EXIT_CODE"
    exit $PHASE2_EXIT_CODE
fi

echo ""

# ============================================================
# Model Upload to Hugging Face Hub
# ============================================================

echo "Uploading Merged Model to Hugging Face Hub"
echo "================================================================================"

UPLOAD_LOG="$LOG_DIR/upload.log"

echo "Log file:    $UPLOAD_LOG"
echo "================================================================================"
echo ""

python3 upload_merged_model.py 2>&1 | tee "$UPLOAD_LOG"
UPLOAD_EXIT_CODE=$?

if [ $UPLOAD_EXIT_CODE -eq 0 ]; then
    echo "✓ Upload completed successfully"
else
    echo "✗ Upload failed with exit code: $UPLOAD_EXIT_CODE"
    exit $UPLOAD_EXIT_CODE
fi

echo ""

# ============================================================
# Final Summary
# ============================================================

echo "================================================================================"
echo "✓ PIPELINE COMPLETE - All tasks executed successfully"
echo "================================================================================"
echo "End time:    $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Log directory: $LOG_DIR"
echo "  - Phase 1 log:  phase1.log"
echo "  - Phase 2 log:  phase2.log"
echo "  - Upload log:   upload.log"
echo ""
echo "To view logs:"
echo "  tail -f $LOG_DIR/phase1.log"
echo "  tail -f $LOG_DIR/phase2.log"
echo "  tail -f $LOG_DIR/upload.log"
echo ""
echo "To view all training sessions:"
echo "  ls -lh logs/"
echo ""
echo "================================================================================"
