"""Evaluation at original resolution."""

from pathlib import Path
import re

import torch
from tqdm import tqdm
import numpy as np
import pandas as pd

from monai.data import DataLoader, decollate_batch
from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    EnsureType,
)

from isles.metrics import (
    compute_dice_f1_instance_difference,
    compute_absolute_volume_difference,
)
from isles.swin.config import SwinTrainConfig
from isles.swin.model import SwinUNETRPredictor
from isles.swin.transforms import get_post_transforms


@torch.no_grad()
def final_evaluation(
    checkpoint_path: Path,
    val_loader: DataLoader,
    config: SwinTrainConfig,
    out_dir: Path,
) -> pd.DataFrame:
    """
    Evaluate model on validation set at original resolution.

    Predictions are made at training resolution, then inverted back to
    original resolution for comparison with original labels.

    Parameters
    ----------
    checkpoint_path : Path
        Path to model checkpoint.
    val_loader : DataLoader
        Validation data loader (should use get_val_transforms with image_only=False).
    config : SwinTrainConfig
        Training configuration.
    out_dir : Path
        Directory where to save final predictions

    Returns
    -------
    pd.DataFrame
        Dataframe with per case results results.
    """
    device = torch.device(config.device)
    predictor = SwinUNETRPredictor.from_checkpoint(
        checkpoint_path=checkpoint_path, device=device, final=True
    )

    out_dir.mkdir(exist_ok=True, parents=True)
    post_transforms = get_post_transforms(config=config, out_dir=out_dir)

    load_label_transforms = Compose(
        [
            LoadImage(image_only=False),
            EnsureChannelFirst(),
            EnsureType(),
        ]
    )

    results = []

    for batch in tqdm(
        val_loader, desc="Running prediction at original resolution", leave=False
    ):
        image = batch["image"].to(device)
        logits = predictor.predict_logits(image)

        batch["pred"] = logits.cpu()
        batch["image"] = batch["image"].cpu()

        batch_list = decollate_batch(batch)

        for sample in batch_list:
            label_path = sample["label"].meta["filename_or_obj"]
            case_id = re.search(r"sub-stroke\d+", label_path).group()
            sample = post_transforms(sample)
            original_label = load_label_transforms(label_path)

            voxel_spacing = np.array(original_label.meta["pixdim"][1:4])
            voxel_size = np.prod(voxel_spacing)

            pred_np = sample["pred"].squeeze().numpy().astype(int)
            label_np = original_label.squeeze().numpy().astype(int)

            # Compute metrics
            abs_vol_diff = compute_absolute_volume_difference(
                label_np, pred_np, voxel_size
            )
            f1_score, instance_count_diff, dice_score = (
                compute_dice_f1_instance_difference(label_np, pred_np)
            )

            results.append(
                {
                    "case_id": case_id,
                    "dice": dice_score,
                    "f1_score": f1_score,
                    "abs_vol_diff": abs_vol_diff,
                    "instance_count_diff": instance_count_diff,
                }
            )

        torch.cuda.empty_cache()
        results_df = pd.DataFrame(results)
        results_df.to_csv(out_dir / "results.csv", index=False)
