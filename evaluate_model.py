#!/usr/bin/env python3
"""
Evaluate model on hawks evaluation dataset (JSONL input + CSV gold labels).

Supports:
- Axolotl LoRA outputs (adapter_model.safetensors + adapter_config.json)
- Fully merged model directories

Evaluates with 3-state metrics: correct, incorrect, blank-correct.

Example:
    python3 evaluate_model.py
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


CSV_FIELDS = [
    "person_name",
    "person_age_at_published",
    "person_address",
    "person_belongs_company",
    "person_position_name",
    "person_violated_law",
    "occured_locations",
    "police_departments",
]


@dataclass
class EvalConfig:
    model_dir: Path = Path("outputs/lora-out-phase2/checkpoint-5000")
    base_model: Optional[str] = "Rakuten/RakutenAI-7B-instruct"
    eval_input: Path = Path("data/test/hawks_eval_input.jsonl")
    eval_gold: Path = Path("data/test/hawks_eval_gold.csv")
    instruction_source: Path = Path("data/train/hawks_train_segments.jsonl")
    output_csv: Path = Path("outputs/eval_predictions.csv")
    output_metrics: Path = Path("outputs/eval_metrics.json")
    max_seq_length: int = 4096
    max_new_tokens: int = 1024
    batch_size: int = 1
    max_samples: Optional[int] = None
    load_in_4bit: bool = False
    torch_dtype: str = "bf16"
    device_map: str = "auto"


def is_adapter_dir(path: Path) -> bool:
    return (path / "adapter_config.json").exists() or (path / "adapter_model.safetensors").exists()


def resolve_base_model(adapter_dir: Path, override: Optional[str]) -> Optional[str]:
    if override:
        return override
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        return None
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return cfg.get("base_model_name_or_path")


def load_model_and_tokenizer(cfg: EvalConfig) -> Tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": cfg.device_map,
    }
    if cfg.load_in_4bit:
        load_kwargs["load_in_4bit"] = True
    else:
        import torch

        if cfg.torch_dtype == "bf16":
            load_kwargs["torch_dtype"] = torch.bfloat16
        elif cfg.torch_dtype == "fp16":
            load_kwargs["torch_dtype"] = torch.float16

    if is_adapter_dir(cfg.model_dir):
        base_model = resolve_base_model(cfg.model_dir, cfg.base_model)
        if not base_model:
            raise ValueError("base_model is required when loading an adapter model.")
        base = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
        try:
            from peft import PeftModel
        except Exception as e:
            raise RuntimeError("peft is required to load LoRA adapters.") from e
        model = PeftModel.from_pretrained(base, str(cfg.model_dir))
        tokenizer_source = base_model
    else:
        model = AutoModelForCausalLM.from_pretrained(str(cfg.model_dir), **load_kwargs)
        tokenizer_source = str(cfg.model_dir)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    return model, tokenizer


def normalize_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def parse_json_object_from_text(text: str) -> Dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    s = text.strip()
    if "</think>" in s:
        s = s.split("</think>", 1)[1].strip()
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = s[start : end + 1]
    try:
        obj = json.loads(snippet)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def extract_offenders(output_obj: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    if not isinstance(output_obj, dict):
        return []
    persons = output_obj.get("persons")
    if not isinstance(persons, list):
        return []
    offenders: List[Dict[str, Any]] = []
    for p in persons:
        if not isinstance(p, dict):
            continue
        is_offender = p.get("is_offender") is True
        act = normalize_value(p.get("act"))
        if is_offender and act:
            offenders.append(p)
    return offenders


def to_csv_like_fields(output_obj: Dict[str, Any] | None) -> Dict[str, str]:
    offenders = extract_offenders(output_obj)
    names = [normalize_value(p.get("name")) for p in offenders]
    ages = [normalize_value(p.get("age")) for p in offenders]
    positions = [normalize_value(p.get("position")) for p in offenders]
    belongs = [normalize_value(p.get("affiliation")) for p in offenders]
    addresses = [normalize_value(p.get("address")) for p in offenders]
    acts = [normalize_value(p.get("act")) for p in offenders]

    def _join_slash(values: List[str]) -> str:
        return "/".join(values) if values else ""

    occured_locations_list = output_obj.get("occured_locations") if isinstance(output_obj, dict) else None
    police_departments_list = output_obj.get("police_departments") if isinstance(output_obj, dict) else None
    occured_locations_csv = (
        ",".join(normalize_value(x) for x in occured_locations_list)
        if isinstance(occured_locations_list, list) and occured_locations_list
        else ""
    )
    police_departments_csv = (
        ",".join(normalize_value(x) for x in police_departments_list)
        if isinstance(police_departments_list, list) and police_departments_list
        else ""
    )

    return {
        "person_name": _join_slash(names),
        "person_age_at_published": _join_slash(ages),
        "person_position_name": _join_slash(positions),
        "person_belongs_company": _join_slash(belongs),
        "person_address": _join_slash(addresses),
        "person_violated_law": _join_slash(acts),
        "occured_locations": occured_locations_csv,
        "police_departments": police_departments_csv,
    }


def load_test_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_instruction_from_segments(path: Path) -> str:
    """Load instruction from segments format JSONL.

    Concatenates segments with label=false.
    """
    if not path.exists():
        return ""

    with path.open("r", encoding="utf-8") as f:
        first_line = f.readline()

    try:
        obj = json.loads(first_line)
    except json.JSONDecodeError:
        return ""

    segments = obj.get("segments", [])
    instruction_parts = []

    for seg in segments:
        if not seg.get("label", False):  # label=false only
            instruction_parts.append(seg.get("text", ""))
        else:
            break  # Stop when label=true

    return "".join(instruction_parts).strip()


def load_jsonl_inputs(path: Path) -> List[Dict[str, str]]:
    """Load evaluation input JSONL.

    Format: {"id": "LVP0214", "body": "..."}
    Output: [{"quality_test_id": "...", "body": "...", "title": ""}, ...]
    """
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            records.append(
                {
                    "quality_test_id": obj["id"],
                    "body": obj["body"],
                    "title": "",  # Eval data has no title
                }
            )
    return records


def compute_detailed_metrics(
    pred_rows: Sequence[Dict[str, str]], gold_rows: Sequence[Dict[str, str]]
) -> Dict[str, Any]:
    """Compute 3-state evaluation metrics.

    States per field:
    - correct: pred == gold and both non-empty
    - incorrect: pred != gold and at least one non-empty
    - blank_match: pred == gold == ""

    blank-vs-blank excluded from denominator for accuracy.
    """
    total = len(gold_rows)
    field_states = {k: {"correct": 0, "incorrect": 0, "blank_match": 0} for k in CSV_FIELDS}
    all_match = 0

    for pred, gold in zip(pred_rows, gold_rows):
        row_all_match = True
        for k in CSV_FIELDS:
            p = normalize_value(pred.get(k))
            g = normalize_value(gold.get(k))

            if p == g:
                if p == "":
                    field_states[k]["blank_match"] += 1
                else:
                    field_states[k]["correct"] += 1
            else:
                field_states[k]["incorrect"] += 1
                row_all_match = False

        if row_all_match:
            all_match += 1

    metrics: Dict[str, Any] = {
        "total_samples": total,
        "all_fields_exact_match": all_match,
        "all_fields_exact_match_rate": all_match / total if total > 0 else 0.0,
    }

    for k in CSV_FIELDS:
        correct = field_states[k]["correct"]
        incorrect = field_states[k]["incorrect"]
        blank = field_states[k]["blank_match"]
        denominator = correct + incorrect  # blank excluded

        metrics[f"{k}/correct"] = correct
        metrics[f"{k}/incorrect"] = incorrect
        metrics[f"{k}/blank_match"] = blank
        metrics[f"{k}/accuracy"] = correct / denominator if denominator > 0 else 0.0
        metrics[f"{k}/total_non_blank"] = denominator

    return metrics


def main() -> None:
    cfg = EvalConfig()

    # Validate inputs
    if not cfg.model_dir.exists():
        raise FileNotFoundError(f"Model dir not found: {cfg.model_dir}")
    if not cfg.eval_input.exists():
        raise FileNotFoundError(f"Eval input not found: {cfg.eval_input}")
    if not cfg.eval_gold.exists():
        raise FileNotFoundError(f"Gold labels not found: {cfg.eval_gold}")

    print(f"\n{'='*80}")
    print("Model Evaluation")
    print(f"{'='*80}")
    print(f"Model dir: {cfg.model_dir}")
    print(f"Eval input: {cfg.eval_input}")
    print(f"Gold labels: {cfg.eval_gold}")

    # 1. Load data
    print("\nLoading data...")
    instruction = load_instruction_from_segments(cfg.instruction_source)
    rows = load_jsonl_inputs(cfg.eval_input)
    gold_rows = load_test_rows(cfg.eval_gold)

    if cfg.max_samples:
        rows = rows[: cfg.max_samples]
        gold_rows = gold_rows[: cfg.max_samples]

    print(f"  Instruction loaded: {len(instruction)} chars")
    print(f"  Eval samples: {len(rows)}")
    print(f"  Gold samples: {len(gold_rows)}")

    if len(rows) != len(gold_rows):
        raise ValueError(f"Sample count mismatch: {len(rows)} vs {len(gold_rows)}")

    # 2. Load model
    print("\nLoading model...")
    model, tokenizer = load_model_and_tokenizer(cfg)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"

    # 3. Build prompts
    print("\nBuilding prompts...")
    prompts: List[str] = []
    for row in rows:
        body = row.get("body") or ""
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": f"Content:{body}"},
        ]
        prompts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    # 4. Run inference
    print(f"\nRunning inference ({len(prompts)} samples, batch_size={cfg.batch_size})...")
    pred_rows: List[Dict[str, str]] = []

    import torch

    with torch.no_grad():
        for start in range(0, len(prompts), cfg.batch_size):
            batch_prompts = prompts[start : start + cfg.batch_size]
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=cfg.max_seq_length,
            )
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            generated = model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
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

            if (start + cfg.batch_size) % 100 == 0 or (start + cfg.batch_size) >= len(prompts):
                progress = min(start + cfg.batch_size, len(prompts))
                print(f"  Progress: {progress}/{len(prompts)}")

    # 5. Write predictions
    print(f"\nWriting predictions...")
    cfg.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with cfg.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["quality_test_id", *CSV_FIELDS])
        writer.writeheader()
        for row, pred in zip(rows, pred_rows):
            out_row = {"quality_test_id": row.get("quality_test_id", "")}
            out_row.update({k: pred.get(k, "") for k in CSV_FIELDS})
            writer.writerow(out_row)

    # 6. Compute metrics
    print(f"\nComputing metrics...")
    gold_csv_rows = [{k: row.get(k, "") for k in CSV_FIELDS} for row in gold_rows]
    metrics = compute_detailed_metrics(pred_rows, gold_csv_rows)

    # 7. Write metrics
    cfg.output_metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))

    # Summary
    print(f"\n{'='*80}")
    print("Results Summary")
    print(f"{'='*80}")
    print(f"Total samples: {metrics['total_samples']}")
    print(f"All fields exact match: {metrics['all_fields_exact_match']}/{metrics['total_samples']}")
    print(f"All fields exact match rate: {metrics['all_fields_exact_match_rate']:.2%}")
    print(f"\nPer-field accuracy (blank-vs-blank excluded from denominator):")
    for k in CSV_FIELDS:
        acc = metrics[f"{k}/accuracy"]
        total_non_blank = metrics[f"{k}/total_non_blank"]
        print(f"  {k}: {acc:.2%} ({total_non_blank} non-blank)")

    print(f"\n{'='*80}")
    print(f"Predictions: {cfg.output_csv}")
    print(f"Metrics: {cfg.output_metrics}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
