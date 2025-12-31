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
RUN_LABEL = "uploaded model"

OUTPUT_CSV = Path("outputs/eval_predictions.csv")
OUTPUT_METRICS = Path("outputs/eval_metrics.json")
OUTPUT_GENERATIONS = Path("outputs/eval_generations.jsonl")
OUTPUT_LOG = Path("outputs/eval_inference.log")

MAX_SEQ_LENGTH = 4096
MAX_NEW_TOKENS = 512
BATCH_SIZE = 8
TORCH_DTYPE = "bf16"
# NOTE: "auto" may place the whole model on a single GPU if it fits.
# "balanced" tries to shard across all visible GPUs.
DEVICE_MAP = "balanced"
LOAD_IN_4BIT = False
SEED = 42

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

    OUTPUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_LOG.open("w", encoding="utf-8") as log_file:

        def log(msg: str) -> None:
            print(msg)
            log_file.write(msg + "\n")
            log_file.flush()

        log(f"\n{'='*80}")
        log(f"Model Evaluation ({RUN_LABEL})")
        log(f"{'='*80}")
        log(f"Model id: {MODEL_ID}")
        log(f"Eval messages: {EVAL_MESSAGES}")
        log(f"Gold labels: {EVAL_GOLD}")

        log("\nLoading data...")
        rows = load_eval_messages(EVAL_MESSAGES)
        gold_rows = load_test_rows(EVAL_GOLD)
        quality_test_ids = [row.get("quality_test_id", "") for row in rows]
        input_texts = [
            "\n".join(m.get("content", "") for m in row.get("messages", []) if m.get("role") == "user")
            for row in rows
        ]

        log(f"  Eval samples: {len(rows)}")
        log(f"  Gold samples: {len(gold_rows)}")
        if len(rows) != len(gold_rows):
            raise ValueError(f"Sample count mismatch: {len(rows)} vs {len(gold_rows)}")

        log("\nLoading model...")
        model, tokenizer = load_model_and_tokenizer()
        model.eval()
        try:
            device_map = getattr(model, "hf_device_map", None)
            if isinstance(device_map, dict):
                used_devices = sorted({str(v) for v in device_map.values()})
                log(f"Devices used: {used_devices}")
        except Exception:
            pass

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"

        log("\nBuilding prompts...")
        prompts: List[str] = []
        for row in rows:
            prompts.append(build_prompt(tokenizer, row["messages"]))

        log(f"\nRunning inference ({len(prompts)} samples, batch_size={BATCH_SIZE})...")
        pred_rows: List[Dict[str, str]] = []
        debug_rows: List[Dict[str, Any]] = []

        import torch
        from transformers import set_seed

        set_seed(SEED)
        eos_token_ids = [tokenizer.eos_token_id]
        end_token_id = tokenizer.convert_tokens_to_ids("<|end|>")
        if (
            end_token_id is not None
            and end_token_id != tokenizer.unk_token_id
            and end_token_id not in eos_token_ids
        ):
            eos_token_ids.append(end_token_id)

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
                    do_sample=True,
                    eos_token_id=eos_token_ids,
                )

                input_len = inputs["input_ids"].shape[1]
                for i in range(len(batch_prompts)):
                    gen_tokens = generated[i][input_len:]
                    text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                    obj = parse_json_object_from_text(text)
                    pred_rows.append(to_csv_like_fields(obj) if obj else {k: "" for k in CSV_FIELDS})

                    quality_test_id = rows[start + i].get("quality_test_id", "")
                    log_file.write(f"\n=== {quality_test_id} ===\n")
                    log_file.write(text + "\n")
                    log_file.flush()

                    debug_rows.append(
                        {
                            "quality_test_id": quality_test_id,
                            "prompt": batch_prompts[i],
                            "generation_text": text,
                            "parsed_ok": bool(obj),
                            "parsed_obj": obj,
                        }
                    )

                if (start + BATCH_SIZE) % 100 == 0 or (start + BATCH_SIZE) >= len(prompts):
                    progress = min(start + BATCH_SIZE, len(prompts))
                    log(f"  Progress: {progress}/{len(prompts)}")

        log(f"\nWriting debug generations ({len(debug_rows)} samples)...")
        OUTPUT_GENERATIONS.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT_GENERATIONS.open("w", encoding="utf-8") as f:
            for row in debug_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        log(f"  Debug generations: {OUTPUT_GENERATIONS}")

        log("\nWriting predictions...")
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
            import csv

            writer = csv.DictWriter(f, fieldnames=["quality_test_id", *CSV_FIELDS])
            writer.writeheader()
            for row, pred in zip(rows, pred_rows):
                out_row = {"quality_test_id": row.get("quality_test_id", "")}
                out_row.update({k: pred.get(k, "") for k in CSV_FIELDS})
                writer.writerow(out_row)

        log("\nComputing metrics...")
        gold_csv_rows = [{k: row.get(k, "") for k in CSV_FIELDS} for row in gold_rows]
        metrics = compute_detailed_metrics(
            pred_rows,
            gold_csv_rows,
            input_texts=input_texts,
            quality_test_ids=quality_test_ids,
        )
        OUTPUT_METRICS.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))

        log(f"\n{'='*80}")
        log("Results Summary")
        log(f"{'='*80}")
        log(f"Total samples: {metrics['total_samples']}")
        log(f"All fields exact match: {metrics['all_fields_exact_match']}/{metrics['total_samples']}")
        log(f"All fields exact match rate: {metrics['all_fields_exact_match_rate']:.2%}")
        log("\nPer-field accuracy (blank-vs-blank excluded from denominator):")
        for k in CSV_FIELDS:
            acc = metrics[f"{k}/accuracy"]
            total_non_blank = metrics[f"{k}/total_non_blank"]
            log(f"  {k}: {acc:.2%} ({total_non_blank} non-blank)")

        if metrics.get("error_breakdown/enabled"):
            log("\nPer-field error breakdown (miss/span/hallucination):")
            examples_by_field = metrics.get("error_breakdown/examples") or {}
            for k in CSV_FIELDS:
                incorrect = int(metrics.get(f"{k}/incorrect", 0))
                miss = int(metrics.get(f"{k}/miss", 0))
                span = int(metrics.get(f"{k}/span_mismatch", 0))
                halluc = int(metrics.get(f"{k}/hallucination", 0))
                other = int(metrics.get(f"{k}/other_incorrect", 0))
                if incorrect == 0:
                    continue
                log(f"  {k}: incorrect={incorrect} miss={miss} span={span} halluc={halluc} other={other}")
                field_examples = examples_by_field.get(k, {})
                for bucket in ("miss", "span_mismatch", "hallucination"):
                    bucket_examples = field_examples.get(bucket) or []
                    if not bucket_examples:
                        continue
                    ids = [e.get("quality_test_id", "") for e in bucket_examples[:3] if e.get("quality_test_id")]
                    if ids:
                        log(f"    {bucket} ex: {', '.join(ids)}")

            age_num = metrics.get("error_breakdown/person_age_at_published_numeric_equivalent_mismatch")
            if isinstance(age_num, int):
                age_incorrect = int(metrics.get("person_age_at_published/incorrect", 0))
                rate = (age_num / age_incorrect) if age_incorrect > 0 else 0.0
                log(f"\nAge numeric-equivalent mismatches: {age_num}/{age_incorrect} ({rate:.2%})")


if __name__ == "__main__":
    main()
