"""
MLflow Logger Callback for Axolotl training.

Logs training metrics to MLflow for experiment tracking and visualization.
"""

import os
from dataclasses import dataclass

from transformers import TrainerCallback


@dataclass
class MLflowLoggerCallbackArgs:
    """No-op args for Axolotl plugin loading."""
    pass


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


class MLflowLoggerCallback(TrainerCallback):
    """
    Callback for logging metrics to MLflow.

    Metrics are logged with a stage prefix for multi-stage training.
    """

    def __init__(self, mlflow_module=None, stage_name: str = "training"):
        """
        Initialize MLflow logger callback.

        Args:
            mlflow_module: The mlflow module (imported in the script using this callback)
            stage_name: Prefix for metric names (e.g., "training", "stage1", "stage2")
        """
        super().__init__()
        if mlflow_module is None:
            try:
                import mlflow as mlflow_module  # type: ignore
            except Exception:
                mlflow_module = None
        self.mlflow = mlflow_module
        self.stage_name = stage_name

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
        return "src.callbacks.mlflow_logger.MLflowLoggerCallbackArgs"

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


class MLflowLoggerPlugin:
    """Axolotl plugin wrapper for MLflowLoggerCallback."""

    def __init__(self, *args, **kwargs):
        pass

    def register(self, cfg):
        return cfg

    def get_input_args(self):
        return "src.callbacks.mlflow_logger.MLflowLoggerCallbackArgs"

    def get_trainer_callbacks(self):
        return [MLflowLoggerCallback]

    def get_callbacks(self):
        return [MLflowLoggerCallback]

    @property
    def trainer_callbacks(self):
        return [MLflowLoggerCallback]

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

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Log metrics to MLflow when training logs are available.

        Args:
            args: Training arguments
            state: Training state
            control: Trainer control
            logs: Dictionary of log values
            **kwargs: Additional arguments

        Returns:
            control: Trainer control (unchanged)
        """
        if not is_main_process():
            return control
        if not logs:
            return control
        if self.mlflow is None:
            return control

        step = int(state.global_step)
        # Filter to numeric metrics only and add stage prefix
        metrics = {
            f"{self.stage_name}/{k}": float(v)
            for k, v in logs.items()
            if isinstance(v, (int, float))
        }

        if metrics:
            self.mlflow.log_metrics(metrics, step=step)

        return control
