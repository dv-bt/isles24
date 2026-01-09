"""
Train multi encoder Swin-UNETR
"""

from pathlib import Path
from dataclasses import asdict
import wandb
from isles.swin.config import SwinTrainConfig
from isles.swin.model import get_model
from isles.swin.training import train_swin, get_swin_dataloaders
from isles.swin.evaluation import final_evaluation
from isles.utils import generate_datalist


def main():
    run_id = "run-018"
    config = SwinTrainConfig(
        model="BaseSwinUNETR",
        max_epochs=500,
        modalities=["cta", "cbf"],
        target_spacing=(1.0, 1.0, 1.0),
        roi_size=(64, 64, 64),
        learning_rate=1e-4,
        crop_ratios=(1, 3),
        include_background=False,
        intensity_windows={
            "cta": [0, 90],
            "cbf": [0, 35],
        },
        batch_size=1,
    )

    data_root = Path("/home/renku/work/data-local")
    pretrained_path = (
        data_root / "pretrained/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt"
    )
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
        dir=run_dir,
        config={
            **asdict(config),
            "loss": "DiceCELoss",
            "optimizer": "AdamW",
            "scheduler": "WarmupCosineSchedule",
        },
        save_code=True,
    )
    artifact = wandb.Artifact("datalist", type="datalist")
    artifact.add_file(run_dir / "datalist.json", name="datalist.json")
    wandb.log_artifact(artifact)

    train_loader, val_loader = get_swin_dataloaders(datalist, config)

    model = get_model(config)
    model.load_pretrained_encoders(pretrained_path)

    train_swin(
        model=model,
        config=config,
        run_dir=run_dir,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    checkpoint_path = run_dir / "checkpoints/best_model.pt"
    eval_dir = run_dir / "evaluation"
    final_evaluation(
        checkpoint_path=checkpoint_path,
        val_loader=val_loader,
        config=config,
        out_dir=eval_dir,
    )
    wandb.save(f"{eval_dir}/**/*", base_path=run_dir)


if __name__ == "__main__":
    main()
