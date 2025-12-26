"""
Generation Evaluation Callback for Axolotl training.

Evaluates generation quality by computing metrics on sampled evaluation data.
Metrics include JSON parsing rate, schema compliance, field matching, and F1 scores.
"""

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from transformers import TrainerCallback


@dataclass
class GenerationEvalCallbackArgs:
    """No-op args for Axolotl plugin loading."""
    pass


# ============================================================================
# Helper functions for evaluation
# ============================================================================


def is_main_process() -> bool:
    """Check if current process is the main process in distributed training."""
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


def distributed_barrier() -> None:
    """Synchronize all processes in distributed training."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except Exception:
        return


def unwrap_model(model):
    """Unwrap model from DataParallel wrapper if needed."""
    return model.module if hasattr(model, "module") else model


def parse_json_object_from_text(text: str) -> Dict[str, Any] | None:
    """
    Parse JSON object from text, handling edge cases like thinking blocks.

    Args:
        text: Text containing JSON object

    Returns:
        Parsed JSON object as dict, or None if parsing fails
    """
    if not isinstance(text, str):
        return None

    s = text.strip()
    # Remove thinking blocks if present
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
    """Normalize value to string for comparison."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def extract_offenders(output_obj: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    """Extract offenders (is_offender=True with act) from output object."""
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
    """Convert output object to CSV-like fields for evaluation."""
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
    """Check if output object has correct schema."""
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
    """Split slash-separated values."""
    parts = [p.strip() for p in normalize_value(value).split("/") if p.strip()]
    return parts


def compute_csv_like_metrics(pred_objs: List[Dict[str, Any] | None], gold_objs: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute CSV-like field metrics for evaluation.

    Computes:
    - JSON parsing rate
    - Schema compliance rate
    - Field-level exact match rates
    - Offender presence F1 score
    - Offender name set F1 score
    - False positive rate on negative samples
    """
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

        # Compute field matches
        if pred_csv is not None:
            for k in csv_keys:
                if normalize_value(pred_csv.get(k)) == normalize_value(gold_csv.get(k)):
                    field_matches[k] += 1
            if all(normalize_value(pred_csv.get(k)) == normalize_value(gold_csv.get(k)) for k in csv_keys):
                all_fields_match += 1

        # Compute offender presence metrics
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

        # Compute offender name set metrics
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


def build_prompt_messages(record: dict) -> List[dict]:
    """Build prompt messages from record (without gold output)."""
    SYSTEM_ROLE = "system"
    USER_ROLE = "user"
    ASSISTANT_ROLE = "assistant"

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
    """
    Run generation evaluation on sample records.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        records: Evaluation records containing 'instruction', 'input', 'output'
        max_seq_length: Maximum sequence length for inputs
        max_samples: Maximum number of samples to evaluate
        seed: Random seed for sampling
        batch_size: Batch size for generation
        max_new_tokens: Maximum tokens to generate

    Returns:
        Dictionary of computed metrics
    """
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


# ============================================================================
# GenerationEvalCallback
# ============================================================================


class GenerationEvalCallback(TrainerCallback):
    """
    Callback for evaluating generation quality during training.

    Samples evaluation data and computes metrics including:
    - JSON parsing rate
    - Schema compliance
    - Field-level matching
    - F1 scores for offender presence and names
    """

    def __init__(
        self,
        *,
        trainer=None,
        eval_records: Sequence[dict] | None = None,
        eval_records_path: str | None = None,
        max_samples: int = 200,
        seed: int = 42,
        batch_size: int = 1,
        max_new_tokens: int = 1024,
        max_seq_length: int = 1536,
    ):
        """
        Initialize GenerationEvalCallback.

        Args:
            trainer: Trainer instance (optional; not provided when loaded as an Axolotl plugin)
            eval_records: Evaluation records with 'instruction', 'input', 'output'
            eval_records_path: Optional JSON/JSONL path to evaluation records
            max_samples: Maximum samples to evaluate per step
            seed: Random seed for sampling
            batch_size: Batch size for generation
            max_new_tokens: Maximum tokens to generate
            max_seq_length: Maximum input sequence length
        """
        super().__init__()
        self.trainer = trainer
        self.eval_records = eval_records
        self.eval_records_path = (
            eval_records_path
            or os.environ.get("GEN_EVAL_RECORDS_PATH")
            or "data/train/hawks_val.json"
        )
        self.max_samples = max_samples
        self.seed = seed
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.max_seq_length = max_seq_length

    def register(self, cfg):
        return cfg

    def get_callbacks(self):
        return [self]

    @property
    def trainer_callbacks(self):
        return [self]

    def get_trainer_callbacks(self):
        return [self]

    def get_input_args(self):
        return "src.callbacks.generation_eval.GenerationEvalCallbackArgs"

    def load_datasets(self, cfg, preprocess=False):
        """No-op dataset loading for Axolotl plugin interface."""
        return None

    def pre_model_load(self, cfg):
        """No-op pre-model-load hook for Axolotl plugin interface."""
        pass

    def post_model_load(self, model, cfg):
        """No-op post-model-load hook for Axolotl plugin interface."""
        pass

    def post_model_build(self, cfg, model):
        """No-op post-model-build hook for Axolotl plugin interface."""
        pass

    def pre_lora_load(self, cfg, model):
        """No-op pre-lora-load hook for Axolotl plugin interface."""
        pass

    def post_lora_load(self, cfg, model):
        """No-op post-lora-load hook for Axolotl plugin interface."""
        pass

    def post_trainer_create(self, cfg, trainer):
        """No-op post-trainer-create hook for Axolotl plugin interface."""
        pass

    def get_trainer_cls(self, cfg):
        """Return None to use default trainer."""
        return None

    def get_training_args(self, cfg):
        """Return None to use default training args."""
        return None

    def get_collator_cls_and_kwargs(self, cfg, is_eval=False):
        """Return None to use default collator."""
        return None

    def create_optimizer(self, cfg, trainer):
        """Return None to use default optimizer."""
        return None

    def create_lr_scheduler(self, cfg, trainer, optimizer, num_training_steps):
        """Return None to use default scheduler."""
        return None

    def add_callbacks_pre_trainer(self, cfg, model):
        """Return empty list for pre-trainer callbacks."""
        return []

    def add_callbacks_post_trainer(self, cfg, trainer):
        """Return empty list for post-trainer callbacks."""
        return []

    def post_train(self, cfg, model):
        """No-op post-train hook for Axolotl plugin interface."""
        pass

    def post_train_unload(self, cfg):
        """No-op post-train-unload hook for Axolotl plugin interface."""
        pass

    def __getattr__(self, name):
        if name.startswith("get_"):
            def _default(*args, **kwargs):
                if name.endswith("_args"):
                    return []
                if name.endswith("_kwargs") or name.endswith("_config") or name.endswith("_updates"):
                    return {}
                if name.endswith("_callbacks"):
                    return []
                return None
            return _default
        raise AttributeError(name)

    def _load_eval_records(self) -> None:
        if self.eval_records is not None:
            return
        if not self.eval_records_path:
            self.eval_records = []
            return
        path = Path(self.eval_records_path)
        if not path.exists():
            self.eval_records = []
            return
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            records = []
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            data = records

        if isinstance(data, dict):
            data = data.get("data") or data.get("records") or []
        if not isinstance(data, list):
            data = []
        self.eval_records = [r for r in data if isinstance(r, dict)]

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """
        Run generation evaluation when evaluation is triggered.

        Computes metrics only on the main process and saves them to file.

        Args:
            args: Training arguments
            state: Training state
            control: Trainer control
            metrics: Current metrics dictionary
            **kwargs: Additional arguments

        Returns:
            control: Trainer control (unchanged)
        """
        self._load_eval_records()
        if not self.eval_records:
            return control

        model = kwargs.get("model")
        tokenizer = kwargs.get("tokenizer")
        if model is None or tokenizer is None:
            if self.trainer is not None:
                model = getattr(self.trainer, "model", None)
                tokenizer = getattr(self.trainer, "tokenizer", None)
        if model is None or tokenizer is None:
            return control

        distributed_barrier()
        if is_main_process():
            gen_metrics = run_generation_eval(
                model=model,
                tokenizer=tokenizer,
                records=self.eval_records,
                max_seq_length=self.max_seq_length,
                max_samples=self.max_samples,
                seed=self.seed,
                batch_size=self.batch_size,
                max_new_tokens=self.max_new_tokens,
            )
            if gen_metrics:
                gen_metrics["gen_global_step"] = float(state.global_step)
                if metrics is not None:
                    metrics.update(gen_metrics)
                if self.trainer is not None and hasattr(self.trainer, "log"):
                    self.trainer.log(gen_metrics)
                out_path = Path(args.output_dir) / f"gen_metrics_step{state.global_step}.json"
                out_path.write_text(json.dumps(gen_metrics, ensure_ascii=False, indent=2))
        distributed_barrier()
        return control
