"""
Train multi encoder Swin-UNETR
"""

from pathlib import Path
from dataclasses import asdict
import wandb
from isles.swin import SwinTrainConfig, MultiEncoderSwinUNETR, train_swin
from isles.utils import generate_datalist
from isles.io import get_dataloader
from isles.transforms import get_train_transforms, get_val_transforms


def main():
    run_id = "run-010"
    config = SwinTrainConfig(
        max_epochs=5,
        modalities=["cta", "cbf"],
        target_spacing=(2.0, 2.0, 2.0),
        roi_size=(64, 64, 64),
    )

    data_root = Path("/home/renku/work/data-local")
    run_dir = data_root / f"runs/{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    datalist = generate_datalist(
        data_root=data_root,
        target_dir=run_dir,
        modalities=config.modalities,
        val_fold=0,
    )

    wandb.init(
        project="ISLES",
        name=run_id,
        config={
            **asdict(config),
            "model": "MultiEncoderSwinUNETR",
        },
    )
    artifact = wandb.Artifact("datalist", type="datalist")
    artifact.add_file(run_dir / "datalist.json", name="datalist.json")
    wandb.log_artifact(artifact)

    # Build data loaders
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

    model = MultiEncoderSwinUNETR(
        modalities=config.modalities,
        feature_size=config.feature_size,
        fusion_kernel_size=config.fusion_kernel_size,
    )

    train_swin(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        orig_loader=orig_loader,
        save_predictions=True,
    )


if __name__ == "__main__":
    main()
