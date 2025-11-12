"""
Generate datalist compatible with MONAI
"""

from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from isles.io import parse_demo_data


def assign_fold(
    data_root: Path, strata_cols: list[str], n_folds: int, random_state: int
) -> pd.DataFrame:
    """Assign a fold to each case, using a stratifield K-fold split with strata_cols
    as targets, and return as dataframe."""

    data = parse_demo_data(data_root)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    data["Stratum_Key"] = data[strata_cols].astype(str).agg("_".join, axis=1)
    y = data["Stratum_Key"]

    for i, (train_index, test_index) in enumerate(skf.split(y, y)):
        data.loc[test_index, "Fold"] = i
    data = data.set_index("Case")
    return data


def main() -> None:
    """Generate datalist compatible with MONAI"""

    # Variables
    data_root = Path("/home/renku/work/data")
    strata_cols = ["Center", "Sex"]
    n_folds = 5
    random_state = 42
    target_dir = data_root / "processed"
    target_dir.mkdir(exist_ok=True, parents=True)

    # Make stratified split
    demo_data = assign_fold(data_root, strata_cols, n_folds, random_state)

    # Build datalist dictionary, using the last fold as testing data
    case_dirs = sorted(data_root.glob("train/derivatives/sub-stroke*"))
    datalist_dict = {"training": [], "testing": []}
    for case_dir in case_dirs:
        case_name = case_dir.name
        path_dict = {
            "image": [
                str(case_dir / f"ses-01/{case_name}_ses-01_space-ncct_cta.nii.gz"),
                str(
                    case_dir
                    / f"ses-01/perfusion-maps/{case_name}_ses-01_space-ncct_cbf.nii.gz"
                ),
                str(
                    case_dir
                    / f"ses-01/perfusion-maps/{case_name}_ses-01_space-ncct_cbv.nii.gz"
                ),
                str(
                    case_dir
                    / f"ses-01/perfusion-maps/{case_name}_ses-01_space-ncct_mtt.nii.gz"
                ),
                str(
                    case_dir
                    / f"ses-01/perfusion-maps/{case_name}_ses-01_space-ncct_tmax.nii.gz"
                ),
            ],
            "label": str(
                case_dir / f"ses-02/{case_name}_ses-02_space-ncct_lesion-msk.nii.gz"
            ),
        }
        fold = demo_data.loc[case_name, "Fold"]
        if fold == n_folds - 1:
            datalist_dict["testing"].append(path_dict)
        else:
            path_dict["fold"] = int(fold)
            datalist_dict["training"].append(path_dict)

    # Save datalist
    with open(target_dir / "datalist-auto3dseg.json", "w") as file:
        json.dump(datalist_dict, file, indent=4)


if __name__ == "__main__":
    main()
