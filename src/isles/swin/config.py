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
    Configuration for multi-encoder Swin-UNETR training.

    Parameters
    ----------
    modalities
        Input modality names (e.g., ["ncct", "cta"]).
    target_spacing
        Target voxel spacing in mm.
    intensity_windows
        Per-modality intensity windows: {"modality": [min, max]}.
        If None, no intensity windowing is applied.
    feature_size
        Swin-UNETR embedding dimension.
    fusion_kernel_size
        Kernel size for multi-encoder fusion convolution.
    num_classes
        Number of classes including background (2 for binary segmentation).
    roi_size
        Patch size for training and sliding window inference.
    batch_size
        Training batch size.
    num_crops_per_image
        Number of patches sampled per image.
    crop_ratios
        Per-class sampling ratios for RandCropByLabelClassesd.
        None for equal sampling.
    max_epochs
        Total training epochs.
    learning_rate
        Initial learning rate for AdamW.
    weight_decay
        Weight decay for AdamW.
    warmup_ratio
        Fraction of training for learning rate warmup.
    amp
        Enable automatic mixed precision.
    include_background
        Include background class in Dice loss/metric.
    val_interval
        Validate every N epochs.
    val_overlap
        Sliding window overlap during training validation.
    val_overlap_final
        Sliding window overlap for final evaluation.
    inferer_batch_size
        Batch size for sliding window inference.
    device
        Device for training ("cuda", "cpu", etc.).
    """

    # Required
    modalities: Sequence[str]

    # Data preprocessing
    target_spacing: Sequence[float] = (1.0, 1.0, 1.0)
    intensity_windows: dict[str, Sequence[float]] | None = None

    # Model architecture
    feature_size: int = 48
    fusion_kernel_size: int = 1
    num_classes: int = 2

    # Training / inference patches
    roi_size: Sequence[int] = (64, 64, 64)
    batch_size: int = 1
    num_crops_per_image: int = 4
    crop_ratios: Sequence[float] | None = None

    # Training
    max_epochs: int = 200
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.05
    amp: bool = True
    include_background: bool = False

    # Validation
    val_interval: int = 5
    val_overlap: float = 0.2
    val_overlap_final: float = 0.5
    inferer_batch_size: int = 4

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
