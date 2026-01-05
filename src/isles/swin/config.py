"""
Configuration classes for training and using Multi-encoder Swin-UNETR
"""

import json
from pathlib import Path
from collections.abc import Sequence
from dataclasses import dataclass, asdict


@dataclass
class SwinTrainConfig:
    """
    Training configuration for multi-encoder Swin-UNETR.

    Parameters
    ----------
    max_epochs : int
        Total training epochs.
    modalities : Sequence of str
        Input modality names (e.g., ["ncct", "cta"]).
    num_classes : int
        Number of classes to predict. This includes background, i.e. 2 for binary
        segmentation.
    target_spacing : Sequence of float
        Target voxel spacing in mm.
    intensity_windows : dict, optional
        Per-modality intensity windows: {"modality": [min, max]}.
        If None, no intensity windowing is applied.
    feature_size : int
        Swin-UNETR feature size (embedding dimension).
    fusion_kernel_size : int
        Kernel size for multi-encoder fusion convolution.
    roi_size : Sequence of int
        Patch size for training and sliding window inference.
    batch_size: int
        Batch size for training.
    inferer_batch_size : int
        Batch size for sliding window inference.
    inferer_overlap : float
        Overlap ratio for sliding window inference (0 to 1).
    inferer_mode : str
        Blending mode for sliding window inference ("gaussian" or "constant").
    learning_rate : float
        Initial learning rate for AdamW.
    weight_decay : float
        Weight decay for AdamW.
    val_interval : int
        Validate every N epochs.
    device : str
        Device to train on ("cuda", "cpu", "cuda:0", etc.).
    amp : bool
        Whether to use automatic mixed precision.
    """

    # Required
    modalities: Sequence[str]

    # Data preprocessing
    target_spacing: Sequence[float] = (2.0, 2.0, 2.0)
    intensity_windows: dict[str, Sequence[float]] | None = None

    # Model architecture
    feature_size: int = 48
    fusion_kernel_size: int = 1
    num_classes: int = 2

    # Training / inference patches
    roi_size: Sequence[int] = (64, 64, 64)
    batch_size: int = 1
    num_crops_per_image: int = 2
    crop_ratios: Sequence[float] | None = None

    # Training
    max_epochs: int = 200
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    amp: bool = True

    # Validation
    val_interval: int = 5
    val_overlap: float = 0.2
    val_overlap_final: float = 0.5
    inferer_batch_size: int = 2

    # Device
    device: str = "cuda"

    def to_json(self, path: str | Path) -> None:
        """Save configuration to JSON file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> "SwinTrainConfig":
        """Load configuration from JSON file."""
        with open(path) as f:
            return cls(**json.load(f))
