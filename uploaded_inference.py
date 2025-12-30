#!/usr/bin/env python3
"""
Run inference from a Hugging Face uploaded model and write evaluation outputs.

Outputs:
  - outputs/eval_predictions.csv
  - outputs/eval_metrics.json
  - outputs/eval_generations.jsonl
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from evaluate_model import (
    CSV_FIELDS,
    build_prompt,
    compute_detailed_metrics,
    load_eval_messages,
    load_test_rows,
    parse_json_object_from_text,
    to_csv_like_fields,
)


MODEL_ID = "teru00801/gpt-oss-20b-person-v1"
EVAL_MESSAGES = Path("data/test/hawks_eval_messages.jsonl")
EVAL_GOLD = Path("data/test/hawks_eval_gold.csv")

OUTPUT_CSV = Path("outputs/eval_predictions.csv")
OUTPUT_METRICS = Path("outputs/eval_metrics.json")
OUTPUT_GENERATIONS = Path("outputs/eval_generations.jsonl")

MAX_SEQ_LENGTH = 4096
MAX_NEW_TOKENS = 4096
BATCH_SIZE = 8
TORCH_DTYPE = "bf16"
DEVICE_MAP = "auto"
LOAD_IN_4BIT = False

HF_TOKEN = (
    os.getenv("HF_AUTH_TOKEN")
    or os.getenv("HF_TOKEN")
    or os.getenv("HUGGINGFACE_HUB_TOKEN")
)


def load_model_and_tokenizer() -> Tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": DEVICE_MAP,
    }
    if HF_TOKEN:
        load_kwargs["token"] = HF_TOKEN

    if LOAD_IN_4BIT:
        load_kwargs["load_in_4bit"] = True
    else:
        import torch

        if TORCH_DTYPE == "bf16":
            load_kwargs["torch_dtype"] = torch.bfloat16
        elif TORCH_DTYPE == "fp16":
            load_kwargs["torch_dtype"] = torch.float16

    model_source = MODEL_ID
    if Path(MODEL_ID).exists():
        model_source = MODEL_ID

    model = AutoModelForCausalLM.from_pretrained(model_source, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True, token=HF_TOKEN)
    return model, tokenizer


def main() -> None:
    if not EVAL_MESSAGES.exists():
        raise FileNotFoundError(f"Eval messages not found: {EVAL_MESSAGES}")
    if not EVAL_GOLD.exists():
        raise FileNotFoundError(f"Gold labels not found: {EVAL_GOLD}")

    print(f"\n{'='*80}")
    print("Model Evaluation (uploaded model)")
    print(f"{'='*80}")
    print(f"Model id: {MODEL_ID}")
    print(f"Eval messages: {EVAL_MESSAGES}")
    print(f"Gold labels: {EVAL_GOLD}")

    print("\nLoading data...")
    rows = load_eval_messages(EVAL_MESSAGES)
    gold_rows = load_test_rows(EVAL_GOLD)

    print(f"  Eval samples: {len(rows)}")
    print(f"  Gold samples: {len(gold_rows)}")
    if len(rows) != len(gold_rows):
        raise ValueError(f"Sample count mismatch: {len(rows)} vs {len(gold_rows)}")

    print("\nLoading model...")
    model, tokenizer = load_model_and_tokenizer()
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"

    print("\nBuilding prompts...")
    prompts: List[str] = []
    for row in rows:
        prompts.append(build_prompt(tokenizer, row["messages"]))

    print(f"\nRunning inference ({len(prompts)} samples, batch_size={BATCH_SIZE})...")
    pred_rows: List[Dict[str, str]] = []
    debug_rows: List[Dict[str, Any]] = []

    import torch

    with torch.no_grad():
        for start in range(0, len(prompts), BATCH_SIZE):
            batch_prompts = prompts[start : start + BATCH_SIZE]
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
            )
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            generated = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

            attention_mask = inputs.get("attention_mask")
            for i in range(len(batch_prompts)):
                prompt_len = (
                    int(attention_mask[i].sum().item())
                    if attention_mask is not None
                    else inputs["input_ids"].shape[1]
                )
                gen_tokens = generated[i][prompt_len:]
                text = tokenizer.decode(gen_tokens, skip_special_tokens=False)
                obj = parse_json_object_from_text(text)
                pred_rows.append(to_csv_like_fields(obj) if obj else {k: "" for k in CSV_FIELDS})
                debug_rows.append(
                    {
                        "quality_test_id": rows[start + i].get("quality_test_id", ""),
                        "prompt": batch_prompts[i],
                        "generation_text": text,
                        "parsed_ok": bool(obj),
                        "parsed_obj": obj,
                    }
                )

            if (start + BATCH_SIZE) % 100 == 0 or (start + BATCH_SIZE) >= len(prompts):
                progress = min(start + BATCH_SIZE, len(prompts))
                print(f"  Progress: {progress}/{len(prompts)}")

    print(f"\nWriting debug generations ({len(debug_rows)} samples)...")
    OUTPUT_GENERATIONS.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_GENERATIONS.open("w", encoding="utf-8") as f:
        for row in debug_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  Debug generations: {OUTPUT_GENERATIONS}")

    print("\nWriting predictions...")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        import csv

        writer = csv.DictWriter(f, fieldnames=["quality_test_id", *CSV_FIELDS])
        writer.writeheader()
        for row, pred in zip(rows, pred_rows):
            out_row = {"quality_test_id": row.get("quality_test_id", "")}
            out_row.update({k: pred.get(k, "") for k in CSV_FIELDS})
            writer.writerow(out_row)

    print("\nComputing metrics...")
    gold_csv_rows = [{k: row.get(k, "") for k in CSV_FIELDS} for row in gold_rows]
    metrics = compute_detailed_metrics(pred_rows, gold_csv_rows)
    OUTPUT_METRICS.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))

    print(f"\n{'='*80}")
    print("Results Summary")
    print(f"{'='*80}")
    print(f"Total samples: {metrics['total_samples']}")
    print(f"All fields exact match: {metrics['all_fields_exact_match']}/{metrics['total_samples']}")
    print(f"All fields exact match rate: {metrics['all_fields_exact_match_rate']:.2%}")
    print("\nPer-field accuracy (blank-vs-blank excluded from denominator):")
    for k in CSV_FIELDS:
        acc = metrics[f"{k}/accuracy"]
        total_non_blank = metrics[f"{k}/total_non_blank"]
        print(f"  {k}: {acc:.2%} ({total_non_blank} non-blank)")


if __name__ == "__main__":
    main()
