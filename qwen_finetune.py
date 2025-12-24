"""
torchrun --nproc_per_node=4 qwen_finetune.py
"""

import gc
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Ensure each torchrun worker sees only its assigned GPU.
if "LOCAL_RANK" in os.environ and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["LOCAL_RANK"]

# Unsloth patches must happen before trl/transformers/peft imports.
import unsloth  # noqa: F401

from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from transformers import TrainerCallback


SYSTEM_ROLE = "system"
USER_ROLE = "user"
ASSISTANT_ROLE = "assistant"

# === Config (edit here) ===
CONFIG = {
    "model_name": "unsloth/Qwen3-30B-A3B-Instruct-2507",
    "load_in_4bit": False,
    "max_seq_length": 4096,
    "per_device_batch_size": 1,
    "gradient_accumulation_steps": 16,  # global batch ~64 if 4 GPUs
    "warmup_ratio": 0.03,
    "logging_steps": 10,
    "save_steps": 200,  # keep aligned with eval_steps
    "save_total_limit": 3,
    "resume_from_checkpoint": True,
    "num_proc": None,  # set >1 to parallelize tokenization
    "eval_path": Path("data/train/hawks_val.json"),
    "eval_steps": 200,
    "eval_generate_max_samples": 200,
    "eval_generate_seed": 42,
    "eval_generate_batch_size": 1,
    "eval_generate_max_new_tokens": 1024,
    "mlflow": {
        "enabled": True,
        "tracking_uri": None,
        "experiment_name": "qwen-finetune",
        "run_name": None,
    },
    "stage1": {
        "data_path": Path("data/train/hawks_train.json"),
        "output_dir": Path("outputs/stage1"),
        "num_train_epochs": 3.0,
        "max_steps": None,
        "learning_rate": 2e-4,
    },
    "stage2": {
        "data_path": Path("data/train/hawks_train_plus_person_dummy_2025-12-23-1817.json"),
        "output_dir": Path("outputs/stage2"),
        "num_train_epochs": None,
        "max_steps": 1500,
        "learning_rate": 1e-4,
    },
}

@dataclass
class StageConfig:
    name: str
    data_path: Path
    output_dir: Path
    num_train_epochs: Optional[float]
    max_steps: Optional[int]
    learning_rate: float
    per_device_batch_size: int
    gradient_accumulation_steps: int
    warmup_ratio: float
    logging_steps: int
    save_steps: int
    save_total_limit: int
    resume_from_checkpoint: bool
    max_seq_length: int = 4096
    bf16: bool = True
    lr_scheduler: str = "cosine"
    num_proc: Optional[int] = None
    eval_steps: Optional[int] = None


def is_main_process() -> bool:
    rank = os.environ.get("RANK")
    if rank is not None:
        try:
            return int(rank) == 0
        except ValueError:
            pass

    try:
        import torch.distributed as dist

        return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
    except Exception:
        return True


def set_device_from_local_rank() -> None:
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        return
    try:
        import torch

        torch.cuda.set_device(int(local_rank))
    except Exception:
        return


def distributed_barrier() -> None:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except Exception:
        return


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def clear_gpu_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def write_config_snapshot(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def load_json_dataset(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_conversation(record: dict) -> List[dict]:
    instruction = (record.get("instruction") or "").strip()
    user_content = (record.get("input") or "").strip()
    assistant_content = (record.get("output") or "").strip()

    messages = []
    if instruction:
        messages.append({"role": SYSTEM_ROLE, "content": instruction})
    messages.append({"role": USER_ROLE, "content": user_content})
    messages.append({"role": ASSISTANT_ROLE, "content": assistant_content})
    return messages


def dataset_from_records(records: Sequence[dict]) -> Dataset:
    conversations = [{"conversations": build_conversation(rec)} for rec in records]
    return Dataset.from_list(conversations)


def format_with_template(dataset: Dataset, tokenizer, num_proc: int | None) -> Dataset:
    def _map_fn(batch):
        convos = batch["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False,
            )
            for convo in convos
        ]
        return {"text": texts}

    map_kwargs = {"batched": True}
    if num_proc is not None:
        map_kwargs["num_proc"] = num_proc
    return dataset.map(_map_fn, **map_kwargs)


def build_prompt_messages(record: dict) -> List[dict]:
    instruction = (record.get("instruction") or "").strip()
    user_content = (record.get("input") or "").strip()
    history = record.get("history") or []

    messages = []
    if instruction:
        messages.append({"role": SYSTEM_ROLE, "content": instruction})

    if isinstance(history, list):
        for turn in history:
            if isinstance(turn, (list, tuple)) and len(turn) == 2:
                user_turn, assistant_turn = turn
                messages.append({"role": USER_ROLE, "content": str(user_turn)})
                messages.append({"role": ASSISTANT_ROLE, "content": str(assistant_turn)})

    messages.append({"role": USER_ROLE, "content": user_content})
    return messages


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


def to_csv_like_fields(output_obj: Dict[str, Any] | None) -> Dict[str, str] | None:
    if not isinstance(output_obj, dict):
        return None

    offenders = extract_offenders(output_obj)
    names = [normalize_value(p.get("name")) for p in offenders]
    ages = [normalize_value(p.get("age")) for p in offenders]
    positions = [normalize_value(p.get("position")) for p in offenders]
    belongs = [normalize_value(p.get("affiliation")) for p in offenders]
    addresses = [normalize_value(p.get("address")) for p in offenders]
    acts = [normalize_value(p.get("act")) for p in offenders]

    def _join_slash(values: List[str]) -> str:
        return "/".join(values) if values else ""

    occured_locations_list = output_obj.get("occured_locations")
    police_departments_list = output_obj.get("police_departments")
    occured_locations_csv = ""
    police_departments_csv = ""
    if offenders:
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


def is_schema_ok(output_obj: Dict[str, Any] | None) -> bool:
    if not isinstance(output_obj, dict):
        return False
    for key in ("persons", "occured_locations", "police_departments"):
        if key not in output_obj:
            return False
    if not isinstance(output_obj.get("persons"), list):
        return False
    if not isinstance(output_obj.get("occured_locations"), list):
        return False
    if not isinstance(output_obj.get("police_departments"), list):
        return False
    return True


def split_slash(value: str) -> List[str]:
    parts = [p.strip() for p in normalize_value(value).split("/") if p.strip()]
    return parts


def compute_csv_like_metrics(pred_objs: List[Dict[str, Any] | None], gold_objs: List[Dict[str, Any]]) -> Dict[str, float]:
    assert len(pred_objs) == len(gold_objs)

    total = len(gold_objs)
    if total == 0:
        return {}

    csv_keys = [
        "person_name",
        "person_age_at_published",
        "person_position_name",
        "person_belongs_company",
        "person_address",
        "person_violated_law",
        "occured_locations",
        "police_departments",
    ]

    parsed = 0
    schema_ok = 0
    field_matches = {k: 0 for k in csv_keys}
    all_fields_match = 0

    tp_presence = fp_presence = fn_presence = tn_presence = 0

    tp_names = fp_names = fn_names = 0
    pred_offender_count = 0
    gold_offender_count = 0

    for pred_obj, gold_obj in zip(pred_objs, gold_objs):
        gold_csv = to_csv_like_fields(gold_obj)
        assert gold_csv is not None

        if pred_obj is not None:
            parsed += 1
        if is_schema_ok(pred_obj):
            schema_ok += 1

        pred_csv = to_csv_like_fields(pred_obj)

        # Strict: if JSON is not parseable, it's always a mismatch (even if gold is empty).
        if pred_csv is not None:
            for k in csv_keys:
                if normalize_value(pred_csv.get(k)) == normalize_value(gold_csv.get(k)):
                    field_matches[k] += 1
            if all(normalize_value(pred_csv.get(k)) == normalize_value(gold_csv.get(k)) for k in csv_keys):
                all_fields_match += 1

        gold_has = bool(normalize_value(gold_csv["person_name"]))
        pred_has = bool(pred_csv is not None and normalize_value(pred_csv["person_name"]))
        if pred_has:
            pred_offender_count += len(split_slash(pred_csv["person_name"]))
        if gold_has:
            gold_offender_count += len(split_slash(gold_csv["person_name"]))

        if gold_has and pred_has:
            tp_presence += 1
        elif (not gold_has) and pred_has:
            fp_presence += 1
        elif gold_has and (not pred_has):
            fn_presence += 1
        else:
            tn_presence += 1

        gold_name_set = set(split_slash(gold_csv["person_name"]))
        pred_name_set = set(split_slash(pred_csv["person_name"])) if pred_csv is not None else set()
        tp_names += len(gold_name_set & pred_name_set)
        fp_names += len(pred_name_set - gold_name_set)
        fn_names += len(gold_name_set - pred_name_set)

    def _safe_div(n: float, d: float) -> float:
        return n / d if d != 0 else 0.0

    presence_precision = _safe_div(tp_presence, tp_presence + fp_presence)
    presence_recall = _safe_div(tp_presence, tp_presence + fn_presence)
    presence_f1 = (
        _safe_div(2 * presence_precision * presence_recall, presence_precision + presence_recall)
        if (presence_precision + presence_recall) != 0
        else 0.0
    )

    name_precision = _safe_div(tp_names, tp_names + fp_names)
    name_recall = _safe_div(tp_names, tp_names + fn_names)
    name_f1 = (
        _safe_div(2 * name_precision * name_recall, name_precision + name_recall)
        if (name_precision + name_recall) != 0
        else 0.0
    )

    metrics: Dict[str, float] = {
        "gen_samples": float(total),
        "gen_json_parse_rate": _safe_div(parsed, total),
        "gen_schema_ok_rate": _safe_div(schema_ok, total),
        "gen_offender_presence_precision": presence_precision,
        "gen_offender_presence_recall": presence_recall,
        "gen_offender_presence_f1": presence_f1,
        "gen_offender_name_set_f1": name_f1,
        "gen_all_fields_exact_match_rate": _safe_div(all_fields_match, total),
        "gen_avg_pred_offender_count": _safe_div(pred_offender_count, total),
        "gen_avg_gold_offender_count": _safe_div(gold_offender_count, total),
        "gen_fp_rate_on_negative": _safe_div(fp_presence, fp_presence + tn_presence),
    }
    for k in csv_keys:
        metrics[f"gen_field_exact_match_rate/{k}"] = _safe_div(field_matches[k], total)
    return metrics


def run_generation_eval(
    *,
    model,
    tokenizer,
    records: Sequence[dict],
    max_seq_length: int,
    max_samples: int,
    seed: int,
    batch_size: int,
    max_new_tokens: int,
) -> Dict[str, float]:
    import torch

    if max_samples <= 0 or batch_size <= 0:
        return {}

    rng = random.Random(seed)
    sample_size = min(max_samples, len(records))
    indices = list(range(len(records)))
    if sample_size < len(indices):
        indices = rng.sample(indices, sample_size)
    sampled = [records[i] for i in indices]

    gold_objs: List[Dict[str, Any]] = []
    prompts: List[str] = []

    for rec in sampled:
        gold_obj = parse_json_object_from_text(rec.get("output", ""))
        if gold_obj is None:
            continue
        gold_objs.append(gold_obj)
        messages = build_prompt_messages(rec)
        prompts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    if not prompts:
        return {}

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    truncation_side_orig = getattr(tokenizer, "truncation_side", "right")
    tokenizer.truncation_side = "left"

    model = unwrap_model(model)
    device = next(model.parameters()).device
    model.eval()

    pred_objs: List[Dict[str, Any] | None] = []
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            attention_mask = inputs.get("attention_mask")
            for i in range(len(batch_prompts)):
                prompt_len = int(attention_mask[i].sum().item()) if attention_mask is not None else inputs["input_ids"].shape[1]
                gen_tokens = generated[i][prompt_len:]
                text = tokenizer.decode(gen_tokens, skip_special_tokens=False)
                pred_objs.append(parse_json_object_from_text(text))

    tokenizer.truncation_side = truncation_side_orig

    # Align lengths (gold_objs can be shorter if we skipped invalid gold outputs).
    pred_objs = pred_objs[: len(gold_objs)]
    return compute_csv_like_metrics(pred_objs, gold_objs)


class MLflowLoggerCallback(TrainerCallback):
    def __init__(self, mlflow_module, stage_name: str):
        super().__init__()
        self.mlflow = mlflow_module
        self.stage_name = stage_name

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return control
        step = int(state.global_step)
        metrics = {f"{self.stage_name}/{k}": float(v) for k, v in logs.items() if isinstance(v, (int, float))}
        if metrics:
            self.mlflow.log_metrics(metrics, step=step)
        return control


class GenerationEvalCallback(TrainerCallback):
    def __init__(
        self,
        *,
        trainer: SFTTrainer,
        eval_records: Sequence[dict],
        max_samples: int,
        seed: int,
        batch_size: int,
        max_new_tokens: int,
        max_seq_length: int,
    ):
        super().__init__()
        self.trainer = trainer
        self.eval_records = eval_records
        self.max_samples = max_samples
        self.seed = seed
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.max_seq_length = max_seq_length

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        distributed_barrier()
        if is_main_process():
            gen_metrics = run_generation_eval(
                model=self.trainer.model,
                tokenizer=self.trainer.tokenizer,
                records=self.eval_records,
                max_seq_length=self.max_seq_length,
                max_samples=self.max_samples,
                seed=self.seed,
                batch_size=self.batch_size,
                max_new_tokens=self.max_new_tokens,
            )
            if gen_metrics:
                gen_metrics["gen_global_step"] = float(state.global_step)
                self.trainer.log(gen_metrics)
                out_path = Path(args.output_dir) / f"gen_metrics_step{state.global_step}.json"
                out_path.write_text(json.dumps(gen_metrics, ensure_ascii=False, indent=2))
        distributed_barrier()
        return control


def load_formatted_dataset(path: Path, tokenizer, num_proc: int | None) -> Dataset:
    records = load_json_dataset(path)
    ds = dataset_from_records(records)
    return format_with_template(ds, tokenizer, num_proc)


def prepare_model(model_name: str, max_seq_length: int, load_in_4bit: bool):
    try:
        from unsloth import FastModel
        from unsloth.chat_templates import get_chat_template
    except NotImplementedError as e:
        raise RuntimeError("Unsloth requires a GPU. Run this script on a CUDA machine.") from e

    # Use FastModel for MOE (and standard dense) models per Unsloth guidance.
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
        load_in_8bit=False,
        full_finetuning=False,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen3-instruct")

    model = FastModel.get_peft_model(
        model,
        r=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=128,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    return model, tokenizer


def build_trainer(model, tokenizer, dataset: Dataset, cfg: StageConfig, eval_dataset: Dataset | None) -> SFTTrainer:
    try:
        from unsloth.chat_templates import train_on_responses_only
    except NotImplementedError as e:
        raise RuntimeError("Unsloth requires a GPU. Run this script on a CUDA machine.") from e

    eval_strategy = "no" if eval_dataset is None else "steps"
    training_args = SFTConfig(
        output_dir=str(cfg.output_dir),
        dataset_text_field="text",
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        max_seq_length=cfg.max_seq_length,
        num_train_epochs=cfg.num_train_epochs if cfg.max_steps is None else None,
        max_steps=cfg.max_steps if cfg.max_steps is not None else -1,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        eval_strategy=eval_strategy,
        eval_steps=cfg.eval_steps if eval_strategy != "no" else None,
        bf16=cfg.bf16,
        lr_scheduler_type=cfg.lr_scheduler,
        optim="adamw_torch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=[],
        packing=False,
        load_best_model_at_end=eval_strategy != "no",
        metric_for_best_model="eval_loss" if eval_strategy != "no" else None,
        greater_is_better=False if eval_strategy != "no" else None,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        dataset_num_proc=cfg.num_proc,
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    return trainer


def resolve_resume_checkpoint(output_dir: Path, resume_from_checkpoint: bool) -> Optional[str]:
    if not resume_from_checkpoint:
        return None
    if not output_dir.exists():
        return None
    try:
        from transformers.trainer_utils import get_last_checkpoint
    except Exception:
        return None
    return get_last_checkpoint(str(output_dir))


def run_stage(
    model,
    tokenizer,
    cfg: StageConfig,
    eval_dataset: Dataset | None,
    eval_records: Sequence[dict] | None,
    eval_generate_cfg: dict,
    mlflow_module=None,
):
    train_ds = load_formatted_dataset(cfg.data_path, tokenizer, cfg.num_proc)
    trainer = build_trainer(model, tokenizer, train_ds, cfg, eval_dataset)
    if eval_dataset is not None and eval_records:
        trainer.add_callback(
            GenerationEvalCallback(
                trainer=trainer,
                eval_records=eval_records,
                max_samples=eval_generate_cfg["max_samples"],
                seed=eval_generate_cfg["seed"],
                batch_size=eval_generate_cfg["batch_size"],
                max_new_tokens=eval_generate_cfg["max_new_tokens"],
                max_seq_length=cfg.max_seq_length,
            )
        )
    if mlflow_module is not None and is_main_process():
        trainer.add_callback(MLflowLoggerCallback(mlflow_module, cfg.name))
    resume_checkpoint = resolve_resume_checkpoint(cfg.output_dir, cfg.resume_from_checkpoint)
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    if eval_dataset is not None:
        trainer.evaluate()
    return unwrap_model(trainer.model)


def main():
    set_device_from_local_rank()
    cfg_global = CONFIG
    stage1_raw = cfg_global["stage1"]
    stage2_raw = cfg_global["stage2"]

    stage1_cfg = StageConfig(
        name="stage1",
        data_path=stage1_raw["data_path"],
        output_dir=stage1_raw["output_dir"],
        num_train_epochs=stage1_raw["num_train_epochs"],
        max_steps=stage1_raw["max_steps"],
        learning_rate=stage1_raw["learning_rate"],
        per_device_batch_size=cfg_global["per_device_batch_size"],
        gradient_accumulation_steps=cfg_global["gradient_accumulation_steps"],
        warmup_ratio=cfg_global["warmup_ratio"],
        logging_steps=cfg_global["logging_steps"],
        save_steps=cfg_global["save_steps"],
        save_total_limit=cfg_global["save_total_limit"],
        resume_from_checkpoint=cfg_global["resume_from_checkpoint"],
        max_seq_length=cfg_global["max_seq_length"],
        num_proc=cfg_global["num_proc"],
        eval_steps=cfg_global["eval_steps"],
    )

    stage2_cfg = StageConfig(
        name="stage2",
        data_path=stage2_raw["data_path"],
        output_dir=stage2_raw["output_dir"],
        num_train_epochs=stage2_raw["num_train_epochs"],
        max_steps=stage2_raw["max_steps"],
        learning_rate=stage2_raw["learning_rate"],
        per_device_batch_size=cfg_global["per_device_batch_size"],
        gradient_accumulation_steps=cfg_global["gradient_accumulation_steps"],
        warmup_ratio=cfg_global["warmup_ratio"],
        logging_steps=cfg_global["logging_steps"],
        save_steps=cfg_global["save_steps"],
        save_total_limit=cfg_global["save_total_limit"],
        resume_from_checkpoint=cfg_global["resume_from_checkpoint"],
        max_seq_length=cfg_global["max_seq_length"],
        num_proc=cfg_global["num_proc"],
        eval_steps=cfg_global["eval_steps"],
    )

    stage1_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    stage2_cfg.output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = prepare_model(
        model_name=cfg_global["model_name"],
        max_seq_length=cfg_global["max_seq_length"],
        load_in_4bit=cfg_global["load_in_4bit"],
    )

    eval_dataset = None
    eval_records = None
    eval_path = cfg_global.get("eval_path")
    if eval_path:
        eval_records = load_json_dataset(eval_path)
        eval_dataset = format_with_template(dataset_from_records(eval_records), tokenizer, cfg_global["num_proc"])

    eval_generate_cfg = {
        "max_samples": cfg_global["eval_generate_max_samples"],
        "seed": cfg_global["eval_generate_seed"],
        "batch_size": cfg_global["eval_generate_batch_size"],
        "max_new_tokens": cfg_global["eval_generate_max_new_tokens"],
    }

    mlflow_cfg = cfg_global.get("mlflow") or {}
    mlflow_module = None
    mlflow_run = None
    if mlflow_cfg.get("enabled") and is_main_process():
        try:
            import mlflow as mlflow_module  # type: ignore
        except Exception as e:
            raise RuntimeError("MLflow is enabled but not installed. Add it to dependencies.") from e
        tracking_uri = mlflow_cfg.get("tracking_uri")
        if tracking_uri:
            mlflow_module.set_tracking_uri(tracking_uri)
        mlflow_module.set_experiment(mlflow_cfg.get("experiment_name") or "qwen-finetune")
        mlflow_run = mlflow_module.start_run(run_name=mlflow_cfg.get("run_name"))
        mlflow_module.log_params(
            {
                "model_name": cfg_global["model_name"],
                "max_seq_length": cfg_global["max_seq_length"],
                "per_device_batch_size": cfg_global["per_device_batch_size"],
                "gradient_accumulation_steps": cfg_global["gradient_accumulation_steps"],
                "warmup_ratio": cfg_global["warmup_ratio"],
                "logging_steps": cfg_global["logging_steps"],
                "save_steps": cfg_global["save_steps"],
                "save_total_limit": cfg_global["save_total_limit"],
                "resume_from_checkpoint": cfg_global["resume_from_checkpoint"],
            }
        )

    model = run_stage(
        model,
        tokenizer,
        stage1_cfg,
        eval_dataset,
        eval_records,
        eval_generate_cfg,
        mlflow_module=mlflow_module,
    )
    if is_main_process():
        stage1_cfg.output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(stage1_cfg.output_dir)
        tokenizer.save_pretrained(stage1_cfg.output_dir)
    clear_gpu_memory()
    model = run_stage(
        model,
        tokenizer,
        stage2_cfg,
        eval_dataset,
        eval_records,
        eval_generate_cfg,
        mlflow_module=mlflow_module,
    )

    if is_main_process():
        stage2_cfg.output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(stage2_cfg.output_dir)
        tokenizer.save_pretrained(stage2_cfg.output_dir)
        write_config_snapshot(stage2_cfg.output_dir / "finetune_config.json", cfg_global)
        if mlflow_module is not None:
            mlflow_module.log_artifact(str(stage2_cfg.output_dir / "finetune_config.json"))
    if mlflow_run is not None and mlflow_module is not None:
        mlflow_module.end_run()


if __name__ == "__main__":
    main()
