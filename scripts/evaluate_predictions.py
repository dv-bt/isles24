"""
Evaluate model predictions at original resolution
"""

from pathlib import Path
import json

from isles.swin.config import SwinTrainConfig
from isles.swin.transforms import get_val_transforms
from isles.swin.training import get_dataloader
from isles.swin.evaluation import final_evaluation


def main():
    data_root = Path("/home/renku/work/data-local")
    run_id = "run-017"
    run_dir = data_root / f"runs/{run_id}"
    out_dir = run_dir / "evaluation"
    checkpoint_path = run_dir / "checkpoints/best_model.pt"

    config = SwinTrainConfig.from_json(run_dir / "config.json")
    with open(run_dir / "datalist.json") as file:
        datalist = json.load(file)

    val_loader = get_dataloader(
        datalist=datalist,
        key="validation",
        transforms=get_val_transforms(config),
        batch_size=config.batch_size,
        cache_rate=0.0,
    )

    final_evaluation(
        checkpoint_path=checkpoint_path,
        val_loader=val_loader,
        config=config,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
