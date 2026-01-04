"""
Code for multi encoder Swin-UNETR
"""

from typing import Any
import copy
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from collections.abc import Sequence

from tqdm import tqdm
import wandb
import pandas as pd
import nibabel as nib

import torch
import torch.nn as nn

from monai.networks.nets.swin_unetr import SwinUNETR, filter_swinunetr
from monai.networks.blocks import UnetrBasicBlock
from monai.networks.utils import copy_model_state
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.data import DataLoader

from isles.io import get_dataloader
from isles.transforms import get_train_transforms, get_val_transforms


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
    max_epochs: int
    modalities: Sequence[str]

    # Data preprocessing
    target_spacing: Sequence[float] = (2.0, 2.0, 2.0)
    intensity_windows: dict[str, Sequence[float]] | None = None

    # Model architecture
    feature_size: int = 48
    fusion_kernel_size: int = 1

    # Training / inference patches
    roi_size: Sequence[int] = (64, 64, 64)
    batch_size: int = 1
    inferer_batch_size: int = 2
    inferer_overlap: float = 0.5
    inferer_mode: str = "gaussian"

    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    # Training loop
    val_interval: int = 1
    device: str = "cuda"
    amp: bool = True

    def to_json(self, path: str | Path) -> None:
        """Save configuration to JSON file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> "SwinTrainConfig":
        """Load configuration from JSON file."""
        with open(path) as f:
            return cls(**json.load(f))


class MultiEncoderSwinUNETR(SwinUNETR):
    """Swin-UNETR with multi encoders.

    Based on <https://arxiv.org/abs/2201.01266>", inherits all the decoder architecture
    from monai.networks.nets.SwinUNETR and replaces the single encoder with per-modality
    encoder and channel fusion.

    The model is for the moment hardcoded to produce binary segmentation restuls.

    Parameters
    ----------
    modalities : list[str]
        List of modality names (e.g., ["CTA", "CBF"]).
    feature_size : int
        Base feature dimension.
    fusion_kernel_size : int
        Kernel size for fusion convolutions.
    **kwargs
        Additional arguments passed to parent SwinUNETR.
    """

    def __init__(
        self,
        modalities: list[str],
        feature_size: int = 48,
        fusion_kernel_size: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(
            in_channels=1,
            feature_size=feature_size,
            out_channels=1,
            **kwargs,
        )

        self.modalities = modalities
        num_modalities = len(modalities)

        self.swin_encoders = nn.ModuleDict(
            {modality: copy.deepcopy(self.swinViT) for modality in modalities}
        )
        del self.swinViT

        self.fusion_layers = nn.ModuleList(
            [
                nn.Conv3d(
                    in_channels=feature_size * mult * num_modalities,
                    out_channels=feature_size * mult,
                    kernel_size=fusion_kernel_size,
                    padding=fusion_kernel_size // 2,
                )
                for mult in [1, 2, 4, 8, 16]
            ]
        )

        # Replace encoder1 to handle multi-channel input
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=kwargs.get("spatial_dims", 3),
            in_channels=num_modalities,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=kwargs.get("norm_name", "instance"),
            res_block=True,
        )

    def load_pretrained_encoders(self, weights_path: str) -> None:
        """Load SSL pretrained weights into all encoders."""
        ssl_weights = torch.load(weights_path, weights_only=False)["state_dict"]

        for modality, encoder in self.swin_encoders.items():
            wrapper = nn.Module()
            wrapper.swinViT = encoder

            _, loaded, _ = copy_model_state(
                wrapper, ssl_weights, filter_func=filter_swinunetr
            )
            print(f"Encoder [{modality}]: loaded {len(loaded)} keys")

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        assert x_in.shape[1] == len(self.modalities)

        # Run each modality through its encoder
        all_hidden_states = [
            self.swin_encoders[modality](x_in[:, i : i + 1, ...], self.normalize)
            for i, modality in enumerate(self.modalities)
        ]

        # Fuse at each scale
        fused_hidden_states = [
            self.fusion_layers[s](torch.cat([hs[s] for hs in all_hidden_states], dim=1))
            for s in range(5)
        ]

        # Decoder (reusing parent's blocks)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(fused_hidden_states[0])
        enc2 = self.encoder3(fused_hidden_states[1])
        enc3 = self.encoder4(fused_hidden_states[2])
        dec4 = self.encoder10(fused_hidden_states[4])

        dec3 = self.decoder5(dec4, fused_hidden_states[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)

        logits = self.out(out)
        return logits


@torch.no_grad()
def predict_volume(
    model: MultiEncoderSwinUNETR,
    image: torch.Tensor,
    roi_size: tuple[int] = (64, 64, 64),
    device: torch.device | str = torch.device("cuda"),
    amp: bool = True,
    sw_batch_size: int = 2,
    overlap: float = 0.5,
    mode: str = "gaussian",
) -> torch.Tensor:
    """
    Run sliding window inference on a single volume.

    Parameters
    ----------
    model : torch.nn.Module
        Segmentation network (outputs logits).
    image : torch.Tensor
        Input volume, shape (1, C, H, W, D).
    roi_size : tuple of int
        Patch size for sliding window.
    device : torch.device | str
        Device to run inference on. Default is 'cuda'
    amp : bool
        Whether to use automatic mixed precision.
    sw_batch_size : int
        Batch size for sliding window inferer. Default is 2.
    overlap : float
        Overlap for sliding window inferer. Default is 0.5.
    mode : str
        Mode for sliding window inferer. Default is 'gaussian'

    Returns
    -------
    torch.Tensor
        Binary mask, shape (1, 1, H, W, D).
    """
    model.to(device)
    model.eval()
    image = image.to(device)

    inferer = SlidingWindowInferer(
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        mode=mode,
    )

    with torch.amp.autocast(device.type, enabled=amp):
        logits = inferer(image, model)

    return (torch.sigmoid(logits) > 0.5).float()


def train_swin(
    model: MultiEncoderSwinUNETR,
    config: SwinTrainConfig,
    run_dir: Path | str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    orig_loader: DataLoader | None = None,
    save_predictions: bool = True,
):
    """
    Train a binary segmentation model.

    Parameters
    ----------
    model : MultiEncoderSwinUNETR
        Segmentation network with out_channels=1.
    config : TrainConfig
        Training configuration.
    run_dir : Path | str
        Directory where to save artifacts.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    orig_loader : DataLoader | None
        Optional validation data loader with the original image spacing.
        This is used at the end of training to calculate the dice score of the best
        model at the original image resolution. If None, this is skipped.
        Default is None.
    save_predictions : bool
        Save predictions of the best model at the original spacing. This is only
        used if orig_loader is not None.
        Default is True.

    Returns
    -------
    dict
        Training summary with best dice and checkpoint path.
    """
    if isinstance(run_dir, str):
        run_dir = Path(run_dir)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Dump config to JSON and log to WandB
    config_path = run_dir / "config.json"
    config.to_json(config_path)
    wandb.save(config_path, base_path=run_dir)

    device = torch.device(config.device)
    model = model.to(device)

    # === Define optimizer, scheduler, losses, etc. ===
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.max_epochs,
    )
    scaler = torch.amp.GradScaler(enabled=config.amp)
    inferer = SlidingWindowInferer(
        roi_size=config.roi_size,
        sw_batch_size=config.inferer_batch_size,
        overlap=config.inferer_overlap,
        mode=config.inferer_mode,
    )

    loss_fn = DiceCELoss(to_onehot_y=False, sigmoid=True)
    dice_metric = DiceMetric(include_background=True, reduction="mean")

    best_dice = 0.0
    best_epoch = 0

    for epoch in range(config.max_epochs):
        # === Training ===
        model.train()
        epoch_loss = 0.0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.max_epochs}")
        for batch in train_pbar:
            image = batch["image"].to(device)
            label = batch["label"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device.type, enabled=config.amp):
                logits = model(image)
                loss = loss_fn(logits, label)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        epoch_loss /= len(train_loader)
        current_lr = scheduler.get_last_lr()[0]

        # === Validation ===
        if (epoch + 1) % config.val_interval == 0 or epoch == config.max_epochs:
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validating", leave=False):
                    image = batch["image"].to(device)
                    label = batch["label"].to(device)

                    with torch.amp.autocast(device.type, enabled=config.amp):
                        logits = inferer(image, model)
                        loss = loss_fn(logits, label)

                    val_loss += loss.item()

                    pred = (torch.sigmoid(logits) > 0.5).float()
                    dice_metric(y_pred=pred, y=label)

            val_loss /= len(val_loader)
            dice = dice_metric.aggregate().item()
            dice_metric.reset()

            # Logging
            metrics = {
                "train/loss": epoch_loss,
                "train/lr": current_lr,
                "val/loss": val_loss,
                "val/dice": dice,
                "epoch": epoch + 1,
            }
            wandb.log(metrics)
            print(
                f"Epoch {epoch + 1}: "
                f"train_loss={epoch_loss:.4f}, "
                f"val_loss={val_loss:.4f}, "
                f"dice={dice:.4f}"
            )

            # == Checkpointing ==
            # Save best dice score
            if dice > best_dice:
                best_dice = dice
                best_epoch = epoch + 1
                checkpoint_path = checkpoint_dir / "best_model.pt"
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_dice": best_dice,
                    },
                    checkpoint_path,
                )
            # Save last model
            if epoch == config.max_epochs:
                checkpoint_path = checkpoint_dir / "last_model.pt"
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "dice": dice,
                    },
                    checkpoint_path,
                )
        else:
            wandb.log(
                {"train/loss": epoch_loss, "train/lr": current_lr, "epoch": epoch + 1}
            )

    wandb.save(checkpoint_dir / "best_model.pt", base_path=run_dir)

    # === Final Evaluation ===
    if orig_loader is not None:
        print(
            "Running final evaluation at original image resolution using the best model"
        )
        eval_dir = run_dir / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = torch.load(checkpoint_dir / "best_model.pt", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        final_results = run_final_evaluation(
            model=model,
            orig_loader=orig_loader,
            config=config,
            output_dir=eval_dir,
            save_predictions=save_predictions,
        )

        wandb.log({"final/dice_mean": final_results["dice_mean"]})
        wandb.log({"final/best_epoch": best_epoch})
        wandb.save(final_results["csv_path"], base_path=run_dir)

        if save_predictions:
            wandb.save(
                f"{final_results['predictions_dir']}/*.nii.gz", base_path=run_dir
            )


@torch.no_grad()
def run_final_evaluation(
    model: "MultiEncoderSwinUNETR",
    orig_loader: DataLoader,
    config: "SwinTrainConfig",
    output_dir: str | Path,
    save_predictions: bool = True,
) -> dict[str, Any]:
    """
    Evaluate model on validation set and save per-case results to CSV.

    Parameters
    ----------
    model : MultiEncoderSwinUNETR
        Trained segmentation network.
    orig_loader : DataLoader
        Validation data loader at the original spacing.
    config : SwinTrainConfig
        Configuration with roi_size, device, amp.
    output_dir : str or Path
        Directory to save CSV results.
    save_predictions : bool
        Whether to save predicted masks as NIfTI files.

    Returns
    -------
    dict
        Mean Dice score, path to CSV file, and path to predictions directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if save_predictions:
        predictions_dir = output_dir / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(config.device)
    model = model.to(device)
    model.eval()

    dice_metric = DiceMetric(include_background=True, reduction="none")
    results = []

    for batch in tqdm(orig_loader, desc="Final evaluation"):
        image = batch["image"].to(device)
        label = batch["label"].to(device)

        pred = predict_volume(model, image, config.roi_size, device, config.amp)
        dice = dice_metric(y_pred=pred, y=label)

        filenames = batch["label_meta_dict"]["filename_or_obj"]
        affines = batch["label_meta_dict"]["affine"]
        scores = dice.squeeze(-1).cpu().tolist()

        for filename, affine, score, pred_single in zip(
            filenames, affines, scores, pred
        ):
            case_id = Path(filename).stem
            results.append({"case_id": case_id, "dice": score})

            if save_predictions:
                mask = pred_single.squeeze(0).cpu().numpy().astype("uint8")
                nib.save(
                    nib.Nifti1Image(mask, affine.cpu().numpy()),
                    predictions_dir / f"{case_id}_pred.nii.gz",
                )

    csv_path = output_dir / "dice_scores.csv"
    results_df = pd.DataFrame(results)
    results_df.to_csv(csv_path, index=False)

    mean_dice = results_df["dice"].mean()
    print(f"Final Dice: {mean_dice:.4f}")

    output = {"dice_mean": mean_dice, "csv_path": str(csv_path)}
    if save_predictions:
        output["predictions_dir"] = str(predictions_dir)

    return output


def get_swin_dataloaders(datalist: dict, config: SwinTrainConfig) -> tuple[DataLoader]:
    """Get dataloader for training multi-encoder Swin-UNETR.

    Parameters
    ----------
    datalist : dict
        Datalist as dictionary. It should have the keys "training" and "validation".
    config : SwinTrainConfig
        Configuration for training multi-encoder Swin-UNETR

    Returns
    -------
    (train_loader, val_loader, orig_loader): tuple[DataLoaders]
        Training, validation, and validation at original spacing dataloaders.
    """
    train_loader = get_dataloader(
        datalist=datalist,
        key="training",
        transforms=get_train_transforms(
            modalitites=config.modalities,
            target_spacing=config.target_spacing,
            roi_size=config.roi_size,
        ),
        batch_size=config.batch_size,
    )

    val_loader = get_dataloader(
        datalist=datalist,
        key="validation",
        transforms=get_val_transforms(
            modalitites=config.modalities,
            target_spacing=config.target_spacing,
        ),
        batch_size=config.batch_size,
    )

    orig_loader = get_dataloader(
        datalist=datalist,
        key="validation",
        transforms=get_val_transforms(
            modalitites=config.modalities,
            target_spacing=None,
        ),
        batch_size=config.batch_size,
    )
    return train_loader, val_loader, orig_loader
