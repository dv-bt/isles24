"""
Train multi encoder Swin-UNETR
"""

from pathlib import Path
from dataclasses import asdict
import wandb
from isles.swin import (
    SwinTrainConfig,
    MultiEncoderSwinUNETR,
    train_swin,
    get_swin_dataloaders,
)
from isles.utils import generate_datalist


def main():
    run_id = "run-010"
    config = SwinTrainConfig(
        max_epochs=5,
        modalities=["cta", "cbf"],
        target_spacing=(2.0, 2.0, 2.0),
        roi_size=(64, 64, 64),
        learning_rate=4e-4,
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
            "loss": "DiceCELoss",
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
        },
        save_code=True,
    )
    artifact = wandb.Artifact("datalist", type="dataset")
    artifact.add_file(run_dir / "datalist.json", name="datalist.json")
    wandb.log_artifact(artifact)

    train_loader, val_loader, orig_loader = get_swin_dataloaders(datalist, config)

    model = MultiEncoderSwinUNETR(
        modalities=config.modalities,
        feature_size=config.feature_size,
        fusion_kernel_size=config.fusion_kernel_size,
    )

    train_swin(
        model=model,
        config=config,
        run_dir=run_dir,
        train_loader=train_loader,
        val_loader=val_loader,
        orig_loader=orig_loader,
        save_predictions=True,
    )


if __name__ == "__main__":
    main()
