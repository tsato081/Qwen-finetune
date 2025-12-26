#!/usr/bin/env python3
"""
Prepare curriculum data for Axolotl training.

This script merges Positive (Hawks) and Negative (Dummy) samples into a balanced dataset.
- Hawks (Positive): 26,221 samples repeated 3x = 78,663 samples
- Dummy (Negative): First 96,000 samples from 100,000 available
- Total: 174,663 samples (~1:1.2 Positive:Negative ratio)

Usage:
    python scripts/prepare_curriculum_data.py
"""

import json
from pathlib import Path

# Paths
HAWKS_TRAIN_PATH = Path("data/train/hawks_train.json")
DUMMY_PATH = Path("data/train/person_dummy_2025-12-23-1817.json")
OUTPUT_PATH = Path("data/train/hawks_train_curriculum.json")

# Hyperparameters
HAWKS_REPETITIONS = 3  # Repeat Hawks 3 times
DUMMY_SUBSET_SIZE = 96000  # Use first 96,000 dummy samples

def main():
    # Check input files exist
    if not HAWKS_TRAIN_PATH.exists():
        raise FileNotFoundError(f"Hawks training file not found: {HAWKS_TRAIN_PATH}")
    if not DUMMY_PATH.exists():
        raise FileNotFoundError(f"Dummy file not found: {DUMMY_PATH}")

    print(f"Loading Hawks training data from {HAWKS_TRAIN_PATH}...")
    with open(HAWKS_TRAIN_PATH, "r") as f:
        hawks_data = json.load(f)
    hawks_count = len(hawks_data)
    print(f"  Loaded {hawks_count} Hawks samples")

    print(f"Loading Dummy data from {DUMMY_PATH}...")
    with open(DUMMY_PATH, "r") as f:
        dummy_data = json.load(f)
    dummy_total = len(dummy_data)
    print(f"  Loaded {dummy_total} Dummy samples (using first {DUMMY_SUBSET_SIZE})")

    # Prepare merged data
    merged_data = []

    # Add Hawks repeated 3x
    print(f"Preparing Hawks data (×{HAWKS_REPETITIONS})...")
    for i in range(HAWKS_REPETITIONS):
        merged_data.extend(hawks_data)
        print(f"  Iteration {i+1}/{HAWKS_REPETITIONS}: {len(merged_data)} total samples")

    hawks_repeated_count = len(merged_data)

    # Add first 96,000 Dummy samples
    print(f"Adding first {DUMMY_SUBSET_SIZE} Dummy samples...")
    merged_data.extend(dummy_data[:DUMMY_SUBSET_SIZE])

    final_count = len(merged_data)

    # Save merged data
    print(f"Saving merged data to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("Data Preparation Summary")
    print("="*60)
    print(f"Hawks (Positive):        {hawks_repeated_count:>10} samples ({hawks_count:>6} ×{HAWKS_REPETITIONS})")
    print(f"Dummy (Negative):        {DUMMY_SUBSET_SIZE:>10} samples")
    print(f"Total:                   {final_count:>10} samples")
    print(f"Ratio (Positive:Negative): 1:{DUMMY_SUBSET_SIZE/hawks_repeated_count:.2f}")
    print(f"Output file:             {OUTPUT_PATH.resolve()}")
    print(f"File size:               {OUTPUT_PATH.stat().st_size / 1024 / 1024:.1f} MB")
    print("="*60)
    print("✅ Data preparation complete!")
    print("\nNote: Axolotl will automatically shuffle this data during training.")
    print("This ensures the model learns to both extract (Hawks) and skip (Dummy) people.")

if __name__ == "__main__":
    main()
