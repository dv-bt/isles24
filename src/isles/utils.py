"""
Utility functions
"""

from typing import Any
from pathlib import Path
import json
import yaml
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from isles.io import parse_demo_data


def _assign_fold(
    data_root: Path,
    strata_cols: list[str],
    n_folds: int,
    random_state: int,
    excluded_cases: list[str],
) -> pd.DataFrame:
    """Assign a fold to each case, using a stratifield K-fold split with strata_cols
    as targets, and return as dataframe."""

    data = parse_demo_data(data_root)
    data = data.loc[~data["Case"].isin(excluded_cases)].reset_index(drop=True)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    data["Stratum_Key"] = data[strata_cols].astype(str).agg("_".join, axis=1)
    y = data["Stratum_Key"]

    for i, (train_index, test_index) in enumerate(skf.split(y, y)):
        data.loc[test_index, "Fold"] = i
    data = data.set_index("Case")
    return data


def _build_path_dict(case_dir: Path, modalities: list[str] | str) -> dict:
    """Build dictionary with for a single case with paths pointing to the right images"""

    if isinstance(modalities, str):
        modalities = [modalities]

    case_name = case_dir.name

    img_paths = []
    for modality in modalities:
        if modality == "ncct":
            img_path = (
                case_dir.parents[1]
                / f"raw_data/{case_name}/ses-01/{case_name}_ses-01_ncct.nii.gz"
            )
        else:
            if modality == "cta":
                img_name = f"ses-01/{case_name}_ses-01_space-ncct_cta.nii.gz"
            else:
                img_name = f"ses-01/perfusion-maps/{case_name}_ses-01_space-ncct_{modality}.nii.gz"
            img_path = case_dir / img_name
        assert img_path.exists()
        img_paths.append(str(img_path))

    path_dict = {
        "image": img_paths,
        "label": str(
            case_dir / f"ses-02/{case_name}_ses-02_space-ncct_lesion-msk.nii.gz"
        ),
    }
    return path_dict


def generate_datalist(
    data_root: Path,
    target_dir: Path,
    modalities: list[str] | str,
    n_folds: int = 5,
    val_fold: int | None = None,
    test_fold: int | None = None,
    strata_cols: list[str] = ["Center", "Sex"],
    random_state: int = 42,
    excluded_cases: list[str] | None = None,
) -> None:
    """Generate datalist compatible with MONAI"""

    target_dir.mkdir(exist_ok=True, parents=True)

    if not excluded_cases:
        excluded_cases = []

    # Make stratified split
    demo_data = _assign_fold(
        data_root, strata_cols, n_folds, random_state, excluded_cases
    )

    # Build datalist dictionary, using the last fold as testing data
    case_dirs = sorted(data_root.glob("train/derivatives/sub-stroke*"))
    datalist_dict = {
        "modalities": modalities,
        "training": [],
        "validation": [],
        "testing": [],
    }
    for case_dir in case_dirs:
        case_name = case_dir.name

        # Ignore excluded cases
        if case_name in excluded_cases:
            print(f"{case_name} excluded")
            continue

        path_dict = _build_path_dict(case_dir, modalities=modalities)
        fold = demo_data.loc[case_name, "Fold"]
        path_dict["fold"] = int(fold)
        if val_fold and val_fold == fold:
            datalist_dict["validation"].append(path_dict)
        elif test_fold and test_fold == fold:
            datalist_dict["testing"].append(path_dict)
        else:
            datalist_dict["training"].append(path_dict)

    # Save datalist
    with open(target_dir / "datalist.json", "w") as file:
        json.dump(datalist_dict, file, indent=4)


def override_swin_params(bundle_dir: Path, params: dict[str, Any]) -> None:
    """Override parameters in swinunetr bundle hyper_parameters.yaml.

    This is useful when training Swin-UNETR via Auto3DSeg and some of the automatically
    set hyperparmeters have to be overridden.

    Parameters
    ----------
    bundle_dir : Path
        Path to the MONAI bundle for the swinunetr model to edit.
    params : dict[str, Any]
        A mapping of 'key':'value' used to update the model hyperparameters.
        'key' should be a valid hyperparameter and 'value' should be a valid value for
        it.
    """
    config_path = bundle_dir / "configs/hyper_parameters.yaml"
    with open(config_path) as f:
        data = yaml.safe_load(f)

    for key, val in params.items():
        data[key] = val

    with open(config_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)
