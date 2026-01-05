"""
Code for multi encoder Swin-UNETR
"""

from pathlib import Path

from tqdm import tqdm
import wandb

import torch
from torch.nn.utils import clip_grad_norm_

from monai.inferers import SlidingWindowInferer
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.data import DataLoader
from monai.optimizers import WarmupCosineSchedule

from isles.io import get_dataloader
from isles.swin.config import SwinTrainConfig
from isles.swin.model import MultiEncoderSwinUNETR
from isles.swin.checkpoint import save_checkpoint
from isles.swin.transforms import (
    get_train_transforms,
    get_val_transforms,
)


def _train_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    config: SwinTrainConfig,
) -> float:
    """
    Run one training epoch.

    Returns
    -------
    float
        Mean loss for the epoch.
    """
    model.train()
    epoch_loss = 0.0
    device = torch.device(config.device)

    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.max_epochs}")
    for batch in train_pbar:
        image = batch["image"].to(device)
        label = batch["label"].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(device.type, enabled=config.amp):
            logits = model(image)
            loss = loss_fn(logits, label)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), max_norm=0.5)
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        train_pbar.set_postfix(loss=f"{loss.item():.4f}")

    return epoch_loss / len(train_loader)


@torch.no_grad()
def _validate_epoch(
    model: torch.nn.Module,
    val_loader: DataLoader,
    loss_fn: torch.nn.Module,
    dice_metric: DiceMetric,
    inferer: SlidingWindowInferer,
    config: SwinTrainConfig,
) -> tuple[float, float]:
    """
    Run one validation epoch.

    Returns
    -------
    tuple[float, float]
        (mean_dice, mean_loss)
    """
    model.eval()
    val_loss = 0.0
    device = torch.device(config.device)

    for batch in tqdm(val_loader, desc="Validating", leave=False):
        image = batch["image"].to(device)
        label = batch["label"].to(device)

        with torch.amp.autocast(device.type, enabled=config.amp):
            logits = inferer(image, model)
            loss = loss_fn(logits, label)

        pred = logits.argmax(dim=1, keepdim=True)
        dice_metric(y_pred=pred, y=label)
        val_loss += loss.item()

    mean_dice = dice_metric.aggregate().item()
    dice_metric.reset()

    return mean_dice, val_loss / len(val_loader)


def train_swin(
    model: MultiEncoderSwinUNETR,
    config: SwinTrainConfig,
    run_dir: Path | str,
    train_loader: DataLoader,
    val_loader: DataLoader,
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
    scheduler = WarmupCosineSchedule(
        optimizer=optimizer,
        t_total=config.max_epochs,
        warmup_steps=int(config.max_epochs * config.warmup_ratio),
        warmup_multiplier=0.1,
    )
    scaler = torch.amp.GradScaler(enabled=config.amp)
    inferer = SlidingWindowInferer(
        roi_size=config.roi_size,
        sw_batch_size=config.inferer_batch_size,
        overlap=config.val_overlap,
        mode="gaussian",
    )

    loss_fn = get_loss_function(config)
    dice_metric = get_dice_metric(config)

    best_dice = -1.0

    for epoch in range(config.max_epochs):
        # === Training ===
        train_loss = _train_epoch(
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            config=config,
        )
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # === Validation ===
        if (epoch + 1) % config.val_interval == 0 or epoch == config.max_epochs:
            val_loss, dice = _validate_epoch(
                model=model,
                val_loader=val_loader,
                loss_fn=loss_fn,
                dice_metric=dice_metric,
                inferer=inferer,
                config=config,
            )
            torch.cuda.empty_cache()

            metrics = {
                "train/loss": train_loss,
                "train/lr": current_lr,
                "val/loss": val_loss,
                "val/dice": dice,
                "epoch": epoch + 1,
            }
            wandb.log(metrics)
            print(
                f"Epoch {epoch + 1}: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, "
                f"dice={dice:.4f}"
            )

            # === Checkpointing ===
            is_best = dice > best_dice
            if is_best:
                best_dice = dice
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                current_dice=dice,
                best_dice=best_dice,
                is_best=is_best,
                config=config,
            )

        else:
            wandb.log(
                {"train/loss": train_loss, "train/lr": current_lr, "epoch": epoch + 1}
            )

    # Upload final best model to WandB
    wandb.save(checkpoint_dir / "best_model.pt", base_path=run_dir)


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
    (train_loader, val_loader): tuple[DataLoaders]
        Training, validation dataloaders.
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

    return train_loader, val_loader

def get_loss_function(config: SwinTrainConfig) -> DiceCELoss:
    """
    Get loss function for softmax segmentation.

    This function could be extended to support different kinds of losses.
    Loss function expects:
        - predictions: [B, C, H, W, D] raw logits
        - labels: [B, 1, H, W, D] integer class indices
    """
    return DiceCELoss(
        include_background=True,
        to_onehot_y=True,
        softmax=True,
        squared_pred=True,
        smooth_nr=0.0,
        smooth_dr=1e-5,
    )


def get_dice_metric(config: SwinTrainConfig) -> DiceMetric:
    """
    Dice metric excluding background class.

    Reports mean Dice across foreground classes only.
    """
    return DiceMetric(
        include_background=False,
        reduction="mean",
        num_classes=config.out_channels,
    )
