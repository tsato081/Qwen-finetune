"""
MLflow Logger Callback for Axolotl training.

Logs training metrics to MLflow for experiment tracking and visualization.
"""

from transformers import TrainerCallback


class MLflowLoggerCallback(TrainerCallback):
    """
    Callback for logging metrics to MLflow.

    Metrics are logged with a stage prefix for multi-stage training.
    """

    def __init__(self, mlflow_module, stage_name: str = "training"):
        """
        Initialize MLflow logger callback.

        Args:
            mlflow_module: The mlflow module (imported in the script using this callback)
            stage_name: Prefix for metric names (e.g., "training", "stage1", "stage2")
        """
        super().__init__()
        self.mlflow = mlflow_module
        self.stage_name = stage_name

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
        if not logs:
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
