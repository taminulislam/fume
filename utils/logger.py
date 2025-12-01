"""
Weights & Biases Logger for FUME
Tracks experiments, metrics, and visualizations
"""

import wandb
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any
import matplotlib.pyplot as plt
from datetime import datetime


class WandBLogger:
    """
    Weights & Biases experiment tracker for FUME

    Features:
    - Automatic experiment logging
    - Metric tracking (train/val)
    - Model checkpoint tracking
    - Visualization logging
    - Confusion matrix logging
    - Hyperparameter tracking
    """

    def __init__(
        self,
        project_name: str = "FUME-Acidosis",
        experiment_name: Optional[str] = None,
        config: Optional[Dict] = None,
        entity: Optional[str] = None,
        resume: bool = False,
        run_id: Optional[str] = None,
        tags: Optional[list] = None
    ):
        """
        Initialize W&B logger

        Args:
            project_name: W&B project name
            experiment_name: Experiment/run name
            config: Configuration dictionary
            entity: W&B entity (username/team)
            resume: Resume from previous run
            run_id: Run ID to resume
            tags: List of tags for the experiment
        """
        self.project_name = project_name
        self.experiment_name = experiment_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Initialize W&B
        if resume and run_id:
            self.run = wandb.init(
                project=project_name,
                name=experiment_name,
                entity=entity,
                id=run_id,
                resume="must",
                tags=tags
            )
        else:
            self.run = wandb.init(
                project=project_name,
                name=experiment_name,
                config=config,
                entity=entity,
                tags=tags
            )

        self.run_id = self.run.id
        print(f"✅ W&B initialized: {project_name}/{self.experiment_name}")
        print(f"   Run ID: {self.run_id}")
        print(f"   View at: {self.run.url}")

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        prefix: str = ""
    ):
        """
        Log metrics to W&B

        Args:
            metrics: Dictionary of metric names and values
            step: Training step/epoch
            prefix: Prefix for metric names (e.g., 'train/', 'val/')
        """
        log_dict = {}
        for key, value in metrics.items():
            log_key = f"{prefix}{key}" if prefix else key
            log_dict[log_key] = value

        if step is not None:
            log_dict['step'] = step

        wandb.log(log_dict, step=step)

    def log_model(
        self,
        model: torch.nn.Module,
        model_name: str = "model",
        aliases: Optional[list] = None
    ):
        """
        Log model architecture to W&B

        Args:
            model: PyTorch model
            model_name: Name for the model
            aliases: List of aliases (e.g., ['latest', 'best'])
        """
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Log model info
        wandb.config.update({
            f"{model_name}_parameters": num_params,
            f"{model_name}_parameters_M": num_params / 1e6
        })

        print(f"✅ Model logged: {model_name} ({num_params/1e6:.2f}M params)")

    def log_checkpoint(
        self,
        checkpoint_path: str,
        metrics: Optional[Dict] = None,
        aliases: Optional[list] = None
    ):
        """
        Log model checkpoint to W&B

        Args:
            checkpoint_path: Path to checkpoint file
            metrics: Dictionary of metrics at this checkpoint
            aliases: List of aliases (e.g., ['latest', 'best'])
        """
        artifact = wandb.Artifact(
            name=f"model-{self.run_id}",
            type="model",
            metadata=metrics
        )
        artifact.add_file(checkpoint_path)

        if aliases is None:
            aliases = ['latest']

        wandb.log_artifact(artifact, aliases=aliases)
        print(f"✅ Checkpoint saved to W&B: {checkpoint_path}")

    def log_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: list,
        title: str = "Confusion Matrix"
    ):
        """
        Log confusion matrix to W&B

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            class_names: List of class names
            title: Plot title
        """
        wandb.log({
            title: wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_true,
                preds=y_pred,
                class_names=class_names
            )
        })

    def log_images(
        self,
        images: Dict[str, Any],
        step: Optional[int] = None,
        caption: Optional[str] = None
    ):
        """
        Log images to W&B

        Args:
            images: Dictionary of image names and image arrays/tensors
            step: Training step
            caption: Caption for images
        """
        log_dict = {}
        for name, img in images.items():
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()

            # Convert to wandb Image
            log_dict[name] = wandb.Image(img, caption=caption)

        wandb.log(log_dict, step=step)

    def log_segmentation(
        self,
        image: np.ndarray,
        pred_mask: np.ndarray,
        true_mask: np.ndarray,
        class_labels: Dict[int, str],
        key: str = "segmentation",
        step: Optional[int] = None
    ):
        """
        Log segmentation results with overlays

        Args:
            image: Input image (H, W) or (H, W, 3)
            pred_mask: Predicted mask (H, W)
            true_mask: Ground truth mask (H, W)
            class_labels: Dict mapping class IDs to names
            key: W&B log key
            step: Training step
        """
        # Create wandb Image with masks
        wandb_img = wandb.Image(
            image,
            masks={
                "predictions": {
                    "mask_data": pred_mask,
                    "class_labels": class_labels
                },
                "ground_truth": {
                    "mask_data": true_mask,
                    "class_labels": class_labels
                }
            }
        )

        wandb.log({key: wandb_img}, step=step)

    def log_histogram(
        self,
        data: np.ndarray,
        name: str,
        step: Optional[int] = None,
        num_bins: int = 64
    ):
        """
        Log histogram to W&B

        Args:
            data: Data array
            name: Histogram name
            step: Training step
            num_bins: Number of histogram bins
        """
        wandb.log({
            name: wandb.Histogram(data, num_bins=num_bins)
        }, step=step)

    def log_table(
        self,
        data: list,
        columns: list,
        table_name: str = "results_table"
    ):
        """
        Log data table to W&B

        Args:
            data: List of rows (each row is a list)
            columns: List of column names
            table_name: Name of the table
        """
        table = wandb.Table(columns=columns, data=data)
        wandb.log({table_name: table})

    def watch_model(
        self,
        model: torch.nn.Module,
        log_freq: int = 100,
        log: str = "all"
    ):
        """
        Watch model gradients and parameters

        Args:
            model: PyTorch model
            log_freq: Logging frequency
            log: What to log ('gradients', 'parameters', 'all')
        """
        wandb.watch(model, log=log, log_freq=log_freq)
        print(f"✅ Watching model (log: {log}, freq: {log_freq})")

    def log_summary(self, summary_dict: Dict[str, Any]):
        """
        Log summary statistics (shown at end of run)

        Args:
            summary_dict: Dictionary of summary statistics
        """
        for key, value in summary_dict.items():
            wandb.run.summary[key] = value

    def log_learning_rate(self, lr: float, step: int):
        """Log current learning rate"""
        wandb.log({"learning_rate": lr}, step=step)

    def log_epoch_time(self, epoch: int, time_seconds: float):
        """Log epoch training time"""
        wandb.log({
            "epoch": epoch,
            "epoch_time_seconds": time_seconds,
            "epoch_time_minutes": time_seconds / 60
        })

    def finish(self):
        """Finish W&B run"""
        wandb.finish()
        print("✅ W&B run finished")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.finish()


class TensorBoardLogger:
    """
    Fallback TensorBoard logger (if W&B not available)
    """

    def __init__(self, log_dir: str = "logs"):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"✅ TensorBoard initialized: {log_dir}")

    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        for key, value in metrics.items():
            log_key = f"{prefix}{key}" if prefix else key
            self.writer.add_scalar(log_key, value, step)

    def log_histogram(self, data: np.ndarray, name: str, step: int):
        self.writer.add_histogram(name, data, step)

    def finish(self):
        self.writer.close()
        print("✅ TensorBoard closed")


def get_logger(use_wandb: bool = True, **kwargs) -> Any:
    """
    Factory function to get appropriate logger

    Args:
        use_wandb: Use W&B (True) or TensorBoard (False)
        **kwargs: Arguments for logger initialization

    Returns:
        Logger instance
    """
    if use_wandb:
        try:
            return WandBLogger(**kwargs)
        except Exception as e:
            print(f"⚠️ W&B initialization failed: {e}")
            print("   Falling back to TensorBoard")
            return TensorBoardLogger(log_dir=kwargs.get('log_dir', 'logs'))
    else:
        return TensorBoardLogger(log_dir=kwargs.get('log_dir', 'logs'))
