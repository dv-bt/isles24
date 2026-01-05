"""
Model checkpoint save and load functions.
"""


from pathlib import Path
from dataclasses import dataclass, asdict
import torch
from isles.swin.config import SwinTrainConfig

@dataclass
class Checkpoint:
    """Container for checkpoint data."""

    epoch: int
    model_state_dict: dict
    optimizer_state_dict: dict
    scheduler_state_dict: dict
    best_dice: float
    current_dice: float | None = None
    config: dict | None = None

    def save(self, path: Path) -> None:
        """Save checkpoint to disk."""
        torch.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.model_state_dict,
                "optimizer_state_dict": self.optimizer_state_dict,
                "scheduler_state_dict": self.scheduler_state_dict,
                "best_dice": self.best_dice,
                "current_dice": self.current_dice,
                "config": self.config,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, device: torch.device | str = "cpu") -> "Checkpoint":
        """Load checkpoint from disk."""
        data = torch.load(path, map_location=device, weights_only=False)
        return cls(
            epoch=data["epoch"],
            model_state_dict=data["model_state_dict"],
            optimizer_state_dict=data["optimizer_state_dict"],
            scheduler_state_dict=data["scheduler_state_dict"],
            best_dice=data["best_dice"],
            current_dice=data.get("current_dice"),
            config=data.get("config"),
        )


def save_checkpoint(
    checkpoint_dir: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    current_dice: float,
    best_dice: float,
    is_best: bool,
    config: SwinTrainConfig,
) -> None:
    """Save training checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = Checkpoint(
        epoch=epoch,
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        scheduler_state_dict=scheduler.state_dict(),
        best_dice=best_dice,
        current_dice=current_dice,
        config=asdict(config),
    )

    checkpoint.save(checkpoint_dir / "last_model.pt")

    if is_best:
        checkpoint.save(checkpoint_dir / "best_model.pt")


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    device: torch.device | str = "cpu",
) -> tuple[int, float]:
    """
    Load checkpoint and restore state.
    
    Returns
    -------
    tuple[int, float]
        (start_epoch, best_dice)
    """
    checkpoint = Checkpoint.load(checkpoint_path, device)
    
    model.load_state_dict(checkpoint.model_state_dict)
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint.optimizer_state_dict)
    
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint.scheduler_state_dict)
    
    return checkpoint.epoch, checkpoint.best_dice
