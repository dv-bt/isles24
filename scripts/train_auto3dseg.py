"""
Train MONAI Auto3DSeg
"""

from pathlib import Path
import yaml
from monai.apps import auto3dseg


def main() -> None:
    """Execute script"""

    # Variables
    WORK_DIR = Path(
        "/home/renku/work/auto3dseg-runs/run-002"
    )
    WORK_DIR.mkdir(exist_ok=True, parents=True)

    datalist_file = Path(
        "/home/renku/work/data-local/processed/datalist-preprocessed.json"
    )
    # All paths are absolute, but dataroot has to be supplied
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
        algos=["swinunetr"]
    )

    # Generate algorithm bundles
    bundle_gen = auto3dseg.BundleGen(
        algo_path=WORK_DIR,
        algos=["swinunetr"],
        data_stats_filename=str(WORK_DIR / "datastats.yaml"),
        data_src_cfg_name=str(WORK_DIR / "input.yaml"),
    )
    bundle_gen.generate(output_folder=WORK_DIR, num_fold=1)
    
    # Update ROI size in swin to smaller values to avoid OOM issues.
    config_path = WORK_DIR / "swinunetr_0/configs/hyper_parameters.yaml"
    with open(config_path) as f:
        data = yaml.safe_load(f)

    data["roi_size"] = [64, 64, 64]

    with open(config_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)

    max_epochs = 100
    train_param = {
        "num_epochs_per_validation": 1,
        "num_images_per_batch": 1,
        "num_epochs": max_epochs,
        "num_warmup_epochs": 1,
        "n_saved": 0,
        "key_metric_n_saved": 1,
        "num_workers": 1,
    }
    runner.set_training_params(train_param)
    runner.run()


if __name__ == "__main__":
    main()
