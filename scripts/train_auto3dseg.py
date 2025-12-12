"""
Train MONAI Auto3DSeg
"""

from pathlib import Path
from dataclasses import dataclass, asdict, field
import yaml
from monai.apps import auto3dseg
from isles.utils import generate_datalist, override_swin_params


@dataclass
class Config:
    run_id: str
    modalities: list[str] | str
    max_epochs: int = 200
    roi_size: list[int, int, int] = field(default_factory=lambda: [64, 64, 64])
    early_stop: bool = False

    def to_yaml(self, path: str | Path) -> None:
        """Write config to yaml"""
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False, default_flow_style=False)

def train_swin(config: Config) -> None:
    """Train Swin-UNETR using auto3dseg"""
    data_root = Path("/home/renku/work/data-local/")
    WORK_DIR = Path(f"/home/renku/work/auto3dseg-runs/{config.run_id}")
    WORK_DIR.mkdir(exist_ok=True, parents=True)
    config.to_yaml(WORK_DIR / "custom_hyperparameters.yaml")

    datalist_file = WORK_DIR / "datalist.json"
    generate_datalist(
        data_root=data_root,
        target_dir=WORK_DIR,
        modalities=config.modalities,
        excluded_cases=["sub-stroke0043"],
    )
    dataroot_dir = ""

    # Run data analyzer separately with 0 workers to avoid shared memory issues.
    datastats_path = WORK_DIR / "datastats.yaml"
    if not datastats_path.exists():
        analyzer = auto3dseg.DataAnalyzer(
            datalist=str(datalist_file),
            dataroot=str(dataroot_dir),
            output_path=str(datastats_path),
            device="cuda",
            worker=0,
        )
        analyzer.get_all_case_stats()

    runner = auto3dseg.AutoRunner(
        work_dir=WORK_DIR,
        input={
            "modality": "CT",
            "datalist": str(datalist_file),
            "dataroot": str(dataroot_dir),
        },
        analyze=False,
        algo_gen=False,
        algos=["swinunetr"],
        ensemble=False,
    )

    # Generate algorithm bundles
    bundle_gen = auto3dseg.BundleGen(
        algo_path=WORK_DIR,
        algos=["swinunetr"],
        data_stats_filename=str(WORK_DIR / "datastats.yaml"),
        data_src_cfg_name=str(WORK_DIR / "input.yaml"),
    )
    bundle_gen.generate(output_folder=WORK_DIR, num_fold=1)
    override_swin_params(
        WORK_DIR / "swinunetr_0",
        {
            "roi_size": config.roi_size,
            "roi_size_valid": config.roi_size,
            "early_stop_mode": config.early_stop,
        },
    )

    train_param = {
        "num_epochs_per_validation": 1,
        "num_images_per_batch": 1,
        "num_epochs": config.max_epochs,
        "num_warmup_epochs": 1,
        "n_saved": 0,
        "key_metric_n_saved": 1,
        "num_workers": 1,
    }
    runner.set_training_params(train_param)
    runner.run()


def main() -> None:
    """Execute script"""

    runs_ids = ["run-009", "run-010"]
    modalities = ["tmax", "mtt"]

    for run_id, modality in zip(runs_ids, modalities):
        config = Config(
            run_id=run_id,
            modalities=modality,
            max_epochs=200,
        )
        train_swin(config)


if __name__ == "__main__":
    main()
