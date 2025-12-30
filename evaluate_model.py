#!/usr/bin/env python3
"""
Evaluate model on Hawks eval dataset (messages JSONL + gold CSV).

Input: data/test/hawks_eval_messages.jsonl
Gold:  data/test/hawks_eval_gold.csv
Output:
  - outputs/eval_predictions.csv
  - outputs/eval_metrics.json
  - outputs/eval_generations.jsonl
  - outputs/eval_inference.log
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
    model_dir: Path = Path("outputs/gpt-oss-out")
    base_model: Optional[str] = "openai/gpt-oss-20b"
    eval_messages: Path = Path("data/test/hawks_eval_messages.jsonl")
    eval_gold: Path = Path("data/test/hawks_eval_gold.csv")
    output_csv: Path = Path("outputs/eval_predictions.csv")
    output_metrics: Path = Path("outputs/eval_metrics.json")
    debug_generations_jsonl: Path = Path("outputs/eval_generations.jsonl")
    output_log: Path = Path("outputs/eval_inference.log")
    max_seq_length: int = 4096
    max_new_tokens: int = 4096
    batch_size: int = 8
    load_in_4bit: bool = False
    torch_dtype: str = "bf16"
    device_map: str = "auto"


def is_adapter_dir(path: Path) -> bool:
    return (path / "adapter_config.json").exists() or (path / "adapter_model.safetensors").exists()


def resolve_model_dir(requested: Path) -> Path:
    if requested.exists():
        merged = requested / "merged"
        if merged.exists():
            return merged
        return requested

    out_dir = Path("outputs/gpt-oss-out")
    merged = out_dir / "merged"
    if merged.exists():
        return merged
    if out_dir.exists():
        return out_dir

    ckpts = sorted(out_dir.glob("checkpoint-*"), key=lambda p: p.name)
    if ckpts:
        return ckpts[-1]

    return requested


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
            raise ValueError("base_model is required when loading a LoRA adapter.")
        base = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
        try:
            from peft import PeftModel
        except Exception as exc:
            raise RuntimeError("peft is required to load LoRA adapters.") from exc
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


def load_eval_messages(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            messages = obj.get("messages", [])
            if not isinstance(messages, list):
                continue
            filtered = [m for m in messages if m.get("role") != "assistant"]
            records.append(
                {
                    "quality_test_id": obj.get("id", ""),
                    "messages": filtered,
                }
            )
    return records


def load_test_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def build_prompt(tokenizer: Any, messages: List[Dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Fallback when chat_template is missing.
    system = ""
    user = ""
    for m in messages:
        if m.get("role") == "system":
            system = m.get("content", "")
        if m.get("role") == "user":
            user = m.get("content", "")
    return f"{system}\n\n{user}\n"


def compute_detailed_metrics(
    pred_rows: Sequence[Dict[str, str]], gold_rows: Sequence[Dict[str, str]]
) -> Dict[str, Any]:
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
        denominator = correct + incorrect
        metrics[f"{k}/correct"] = correct
        metrics[f"{k}/incorrect"] = incorrect
        metrics[f"{k}/blank_match"] = blank
        metrics[f"{k}/accuracy"] = correct / denominator if denominator > 0 else 0.0
        metrics[f"{k}/total_non_blank"] = denominator

    return metrics


def main() -> None:
    cfg = EvalConfig()
    cfg.model_dir = resolve_model_dir(cfg.model_dir)

    if not cfg.model_dir.exists():
        raise FileNotFoundError(
            f"Model dir not found: {cfg.model_dir}\n"
            "Tip: ensure outputs/gpt-oss-out exists or set EvalConfig.model_dir."
        )
    if not cfg.eval_messages.exists():
        raise FileNotFoundError(f"Eval messages not found: {cfg.eval_messages}")
    if not cfg.eval_gold.exists():
        raise FileNotFoundError(f"Gold labels not found: {cfg.eval_gold}")

    cfg.output_log.parent.mkdir(parents=True, exist_ok=True)
    with cfg.output_log.open("w", encoding="utf-8") as log_file:

        def log(msg: str) -> None:
            print(msg)
            log_file.write(msg + "\n")
            log_file.flush()

        log(f"\n{'='*80}")
        log("Model Evaluation (messages)")
        log(f"{'='*80}")
        log(f"Model dir: {cfg.model_dir}")
        log(f"Eval messages: {cfg.eval_messages}")
        log(f"Gold labels: {cfg.eval_gold}")

        log("\nLoading data...")
        rows = load_eval_messages(cfg.eval_messages)
        gold_rows = load_test_rows(cfg.eval_gold)

        log(f"  Eval samples: {len(rows)}")
        log(f"  Gold samples: {len(gold_rows)}")
        if len(rows) != len(gold_rows):
            raise ValueError(f"Sample count mismatch: {len(rows)} vs {len(gold_rows)}")

        log("\nLoading model...")
        model, tokenizer = load_model_and_tokenizer(cfg)
        model.eval()

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.truncation_side = "left"

        log("\nBuilding prompts...")
        prompts: List[str] = []
        for row in rows:
            prompts.append(build_prompt(tokenizer, row["messages"]))

        log(f"\nRunning inference ({len(prompts)} samples, batch_size={cfg.batch_size})...")
        pred_rows: List[Dict[str, str]] = []
        debug_rows: List[Dict[str, Any]] = []

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

                if (start + cfg.batch_size) % 100 == 0 or (start + cfg.batch_size) >= len(prompts):
                    progress = min(start + cfg.batch_size, len(prompts))
                    log(f"  Progress: {progress}/{len(prompts)}")

        log(f"\nWriting debug generations ({len(debug_rows)} samples)...")
        cfg.debug_generations_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with cfg.debug_generations_jsonl.open("w", encoding="utf-8") as f:
            for row in debug_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        log(f"  Debug generations: {cfg.debug_generations_jsonl}")

        log("\nWriting predictions...")
        cfg.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with cfg.output_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["quality_test_id", *CSV_FIELDS])
            writer.writeheader()
            for row, pred in zip(rows, pred_rows):
                out_row = {"quality_test_id": row.get("quality_test_id", "")}
                out_row.update({k: pred.get(k, "") for k in CSV_FIELDS})
                writer.writerow(out_row)

        log("\nComputing metrics...")
        gold_csv_rows = [{k: row.get(k, "") for k in CSV_FIELDS} for row in gold_rows]
        metrics = compute_detailed_metrics(pred_rows, gold_csv_rows)
        cfg.output_metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))

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


if __name__ == "__main__":
    main()
