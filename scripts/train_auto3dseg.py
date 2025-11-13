"""
Train MONAI Auto3DSeg
"""

from pathlib import Path
from datetime import datetime
from monai.apps import auto3dseg


def main() -> None:
    """Execute script"""

    # Variables
    WORK_DIR = Path(
        f"/home/renku/work/auto3dseg-runs/auto3dseg-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    WORK_DIR.mkdir(exist_ok=True)

    datalist_file = Path(
        "/home/renku/work/data-local/processed/datalist-auto3dseg.json"
    )
    dataroot_dir = Path("/home/renku/work/data-local/train")

    # Run data analyzer separately with 0 workers to avoid shared memory issues.
    analyzer = auto3dseg.DataAnalyzer(
        datalist=str(datalist_file),
        dataroot=str(dataroot_dir),
        output_path=str(WORK_DIR / "datastats.yaml"),
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
    )

    # Generate algorithm bundles
    bundle_gen = auto3dseg.BundleGen(
        algo_path=WORK_DIR,
        algos=["segresnet", "swinunetr"],
        data_stats_filename=str(WORK_DIR / "datastats.yaml"),
        data_src_cfg_name=str(WORK_DIR / "input.yaml"),
    )

    # Override hardcoded persistent_workers=True in bunlde training algorithms
    algo_template_dir = WORK_DIR / "algorithm_templates"
    script_paths = list(algo_template_dir.rglob("train.py")) + list(
        algo_template_dir.rglob("segmenter.py")
    )
    find = "persistent_workers=True"
    replace = "persistent_workers=False"
    for script_path in script_paths:
        with open(script_path, "r") as f:
            content = f.read()
        if find in content:
            content = content.replace(find, replace)
            with open(script_path, "w") as f:
                f.write(content)
            print(f"Patched: {script_path}")
        else:
            print(f"Text not found or already patched: {script_path}")

    # Generate algorithm using the correct parameters
    bundle_gen.generate(output_folder=WORK_DIR)

    max_epochs = 100
    train_param = {
        "num_epochs_per_validation": 1,
        "num_images_per_batch": 1,
        "num_epochs": max_epochs,
        "num_warmup_epochs": 1,
        "n_saved": 0,
        "key_metric_n_saved": 1,
        "num_workers": 0,
    }
    runner.set_training_params(train_param)
    runner.run()


if __name__ == "__main__":
    main()
