#!/usr/bin/env python3
"""
Run baseline inference with the base model (openai/gpt-oss-20b).
Uses the same settings as uploaded_inference.py, with separate outputs.
"""

from pathlib import Path

import uploaded_inference as runner


def main() -> None:
    runner.MODEL_ID = "openai/gpt-oss-20b"
    runner.RUN_LABEL = "baseline model"
    runner.OUTPUT_CSV = Path("outputs/baseline/eval_predictions.csv")
    runner.OUTPUT_METRICS = Path("outputs/baseline/eval_metrics.json")
    runner.OUTPUT_GENERATIONS = Path("outputs/baseline/eval_generations.jsonl")
    runner.OUTPUT_LOG = Path("outputs/baseline/eval_inference.log")
    runner.main()


if __name__ == "__main__":
    main()
