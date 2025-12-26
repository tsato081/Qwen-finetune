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
