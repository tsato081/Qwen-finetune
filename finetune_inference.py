"""
Run inference on the test CSV and export predictions + metrics.

Supports:
- Axolotl LoRA outputs (adapter_model.safetensors + adapter_config.json)
- Fully merged model directories

Example:
    python3 finetune_inference.py
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
class InferenceConfig:
    model_dir: Path = Path("outputs/axolotl_qwen_finetune")
    base_model: Optional[str] = None
    test_csv: Path = Path("data/test/Hawks_正解データマスター - ver 5.0 csv出力用.csv")
    instruction_source: Path = Path("data/train/hawks_train.json")
    output_csv: Path = Path("outputs/test_predictions.csv")
    output_metrics: Path = Path("outputs/test_metrics.json")
    max_seq_length: int = 1536
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


def load_model_and_tokenizer(cfg: InferenceConfig) -> Tuple[Any, Any]:
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


def load_instruction(path: Path) -> str:
    if not path.exists():
        return ""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data:
        return ""
    return (data[0].get("instruction") or "").strip()


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


def normalize_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


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


def build_prompt_text(instruction: str, title: str, body: str) -> List[Dict[str, str]]:
    user_content = f"Content:{title}{body}"
    messages = []
    if instruction:
        messages.append({"role": "system", "content": instruction})
    messages.append({"role": "user", "content": user_content})
    return messages


def compute_csv_metrics(pred_rows: Sequence[Dict[str, str]], gold_rows: Sequence[Dict[str, str]]) -> Dict[str, float]:
    assert len(pred_rows) == len(gold_rows)
    total = len(gold_rows)
    if total == 0:
        return {}

    field_matches = {k: 0 for k in CSV_FIELDS}
    all_fields_match = 0

    for pred, gold in zip(pred_rows, gold_rows):
        if all(normalize_value(pred.get(k)) == normalize_value(gold.get(k)) for k in CSV_FIELDS):
            all_fields_match += 1
        for k in CSV_FIELDS:
            if normalize_value(pred.get(k)) == normalize_value(gold.get(k)):
                field_matches[k] += 1

    metrics: Dict[str, float] = {
        "samples": float(total),
        "all_fields_exact_match_rate": all_fields_match / total,
    }
    for k in CSV_FIELDS:
        metrics[f"field_exact_match_rate/{k}"] = field_matches[k] / total
    return metrics


def main() -> None:
    cfg = InferenceConfig()

    if not cfg.model_dir.exists():
        raise FileNotFoundError(f"Model dir not found: {cfg.model_dir}")
    if not cfg.test_csv.exists():
        raise FileNotFoundError(f"Test CSV not found: {cfg.test_csv}")

    instruction = load_instruction(cfg.instruction_source)
    rows = load_test_rows(cfg.test_csv)
    if cfg.max_samples:
        rows = rows[: cfg.max_samples]

    model, tokenizer = load_model_and_tokenizer(cfg)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    truncation_side_orig = getattr(tokenizer, "truncation_side", "right")
    tokenizer.truncation_side = "left"

    prompts: List[str] = []
    for row in rows:
        title = row.get("title") or ""
        body = row.get("body") or ""
        messages = build_prompt_text(instruction, title, body)
        prompts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    pred_rows: List[Dict[str, str]] = []
    raw_outputs: List[str] = []

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
                raw_outputs.append(text)
                obj = parse_json_object_from_text(text)
                pred_rows.append(to_csv_like_fields(obj) if obj is not None else {k: "" for k in CSV_FIELDS})

    tokenizer.truncation_side = truncation_side_orig

    output_dir = cfg.output_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with cfg.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["quality_test_id", *CSV_FIELDS])
        writer.writeheader()
        for row, pred in zip(rows, pred_rows):
            out_row = {"quality_test_id": row.get("quality_test_id", "")}
            out_row.update({k: pred.get(k, "") for k in CSV_FIELDS})
            writer.writerow(out_row)

    gold_rows = [
        {k: row.get(k, "") for k in CSV_FIELDS}
        for row in rows
    ]
    metrics = compute_csv_metrics(pred_rows, gold_rows)
    cfg.output_metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))

    print(f"wrote predictions: {cfg.output_csv}")
    print(f"wrote metrics: {cfg.output_metrics}")


if __name__ == "__main__":
    main()
