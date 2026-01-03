"""
Utility functions
"""

import logging
from typing import Any, Iterable, Generator
from pathlib import Path
import json
import yaml
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.model_selection import StratifiedKFold
from isles.io import parse_demo_data
from isles.logging import operation_logger


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
        fold = int(demo_data.loc[case_name, "Fold"])
        path_dict["fold"] = fold
        if val_fold is not None and (val_fold == fold):
            datalist_dict["validation"].append(path_dict)
        elif test_fold is not None and (test_fold == fold):
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


def to_paths(
    image_entry: str | Iterable[str], dataroot: Path = Path(".")
) -> Generator[Path, None, None]:
    """
    Yield Path objects for each image path in a datalist entry.

    Parameters
    ----------
    image_entry : str or Iterable[str]
        A single image path string or an iterable of path strings.
    dataroot : Path, optional
        The root directory to prepend to relative paths.
        Default is the current working directory Path(".").

    Yields
    ------
    Path
        The resolved Path object. If the input path is absolute, it is yielded
        as-is; otherwise, it is joined with `dataroot`.
    """
    sources = [image_entry] if isinstance(image_entry, str) else image_entry

    for source in sources:
        path = Path(source)
        yield path if path.is_absolute() else dataroot / path


def _snap_affines_image(
    channel_list: Iterable[Path],
    modalities: list[str],
    rtol: float = 1e-5,
    atol: float = 1e-6,
    logger: logging.Logger | None = None,
) -> None:
    """Snap affine transforms if they're closer than a relative tolerance."""

    if len(channel_list) <= 1:
        return

    # Use ncct channel as reference
    idx_ncct = modalities.index("ncct")
    ref_img = nib.load(channel_list[idx_ncct])
    ref_affine = ref_img.affine
    ref_shape = ref_img.shape

    for idx, img_path in enumerate(channel_list):
        if idx == idx_ncct:
            continue

        img = nib.load(img_path)
        affine = img.affine
        shape = img.shape
        changed = False

        assert shape == ref_shape

        if np.allclose(affine, ref_affine, rtol=rtol, atol=atol):
            data = img.get_fdata(dtype=img.get_data_dtype())
            header = img.header.copy()
            header.set_qform(ref_affine)
            header.set_sform(ref_affine)
            new_img = nib.Nifti1Image(data, ref_affine, header)
            nib.save(new_img, img_path)
            changed = True

        if logger is not None:
            if changed:
                logger.info(f"{img_path.name} affine snapped to NCCT")
            else:
                logger.warning(
                    f"{img_path.name} differs by more than rtol={rtol} and atol={atol}"
                    "from ncct. "
                    "Affine was not snapped, consider re-registering images."
                )


def snap_affines(
    data_root: Path,
    modalities: list[str] | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-6,
    log_file: Path | None = None,
) -> None:
    """Snap image affine transforms if they're closer than a given tolerance.

    Images will be snapped to the NCCT affine. This is necessary to have MONAI
    trasnforms treat each case as a multi-channel image.
    """

    if modalities is None:
        modalities = ["cta", "cbv", "cbf", "mtt", "tmax"]

    case_dirs = sorted(data_root.glob("train/derivatives/sub-stroke*"))

    with operation_logger("affine_snap_logger", log_file=log_file) as logger:
        for case_dir in case_dirs:
            path_dict = _build_path_dict(case_dir, modalities=modalities)
            channel_list = to_paths(path_dict["image"])
            _snap_affines_image(
                channel_list, modalities, rtol=rtol, atol=atol, logger=logger
            )
