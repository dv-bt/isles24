"""
Utility functions
"""

import re
import logging
from typing import Any
from collections.abc import Iterable, Iterator
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


def _get_image_path(case_dir: Path, modality: str) -> str:
    """Get image path for a given case and modality or label.

    Parameters
    ----------
    case_dir : Path
        Full path to the case directory
    modality : str
        Modality to access. If set to "label", the function will return the lesion masks
        instead.

    Returns
    -------
    image_path : str
        The full path to the image as a string

    Raises
    ------
    AssertionError
        If the path doesn't exist
    """

    case_name = case_dir.name
    perfusion_maps = ["cbf", "cbv", "mtt", "tmax"]

    paths = {
        "ncct": (
            case_dir.parents[1]
            / f"raw_data/{case_name}/ses-01/{case_name}_ses-01_ncct.nii.gz"
        ),
        "cta": case_dir / f"ses-01/{case_name}_ses-01_space-ncct_cta.nii.gz",
        "label": case_dir / f"ses-02/{case_name}_ses-02_space-ncct_lesion-msk.nii.gz",
        **{
            mod: (
                case_dir
                / f"ses-01/perfusion-maps/{case_name}_ses-01_space-ncct_{mod}.nii.gz"
            )
            for mod in perfusion_maps
        },
    }

    image_path = paths[modality]
    assert image_path.exists()
    return str(image_path)


def _build_path_dict(
    case_dir: Path, modalities: list[str] | str, process_guide: str | None = None
) -> dict[str, str | list[str]]:
    """
    Build dictionary with for a single case with paths pointing to the right images and
    labels.

    Parameters
    ----------
    case_dir : Path
        Full path to the case directory.
    modalities : list[str] | str
        Modalities to consider for the multichannel image.
    process_guide : str | None
        Modality to add as a "guide" key, used as an explicit guide for processing
        steps like foreground crop. If None, the "guide" key will not be added.
        Default is None.

    Returns
    -------
    path_dict : dict
        Dictionary with keys:
        - "image": list of image paths as strings
        - "label": path to the label
        - "guide": path to image to be used as guide for processing, optional.

    Raises
    ------
    AssertionError
        If any of the requested paths does not exist.
    """

    if isinstance(modalities, str):
        modalities = [modalities]

    img_paths = []
    for modality in modalities:
        img_paths.append(_get_image_path(case_dir, modality))

    path_dict = {
        "image": img_paths,
        "label": _get_image_path(case_dir, "label"),
    }

    if process_guide is not None:
        path_dict["guide"] = _get_image_path(case_dir, process_guide)

    return path_dict


def patch_guide(datalist: dict, process_guide: str) -> dict:
    """Patch processing guide if the loaded datalist doesn't have any"""

    if "process_guide" in datalist.keys():
        print(
            "Datalist has already a specified processing guide: "
            f"{datalist['process_guide']}"
        )
        return datalist

    datalist["process_guide"] = process_guide
    for split in ["training", "validation", "testing"]:
        for case in datalist[split]:
            case_dir = re.match(r"(.*)/ses-02/", case["label"]).group(1)
            split["guide"] = _get_image_path(case_dir=case_dir, modality=process_guide)

    print(f"Patched datalist with processing guide: {process_guide}")
    return datalist


def generate_datalist(
    data_root: Path,
    target_dir: Path,
    modalities: list[str] | str,
    process_guide: str | None = None,
    n_folds: int = 5,
    val_fold: int | None = None,
    test_fold: int | None = None,
    strata_cols: list[str] = ["Center", "Sex"],
    random_state: int = 42,
    excluded_cases: list[str] | None = None,
) -> dict:
    """Generate datalist compatible with MONAI and return it as a dictionary"""

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
    if process_guide is not None:
        datalist_dict["process_guide"] = process_guide

    for case_dir in case_dirs:
        case_name = case_dir.name

        # Ignore excluded cases
        if case_name in excluded_cases:
            print(f"{case_name} excluded")
            continue

        path_dict = _build_path_dict(
            case_dir, modalities=modalities, process_guide=process_guide
        )

        # Assign case to train, test, or validation split.
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
    return datalist_dict


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
) -> Iterator[Path]:
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
    reference_idx: int,
    atol: float = 1e-4,
    logger: logging.Logger | None = None,
    dry_run: bool = False,
) -> None:
    """Snap affine transforms if they're closer than a given tolerance."""

    # Use ncct channel as reference
    ref_img = nib.load(channel_list[reference_idx])
    ref_affine = ref_img.affine

    for idx, img_path in enumerate(channel_list):
        if idx == reference_idx:
            continue

        img = nib.load(img_path)

        if not np.allclose(img.affine, ref_affine, atol=atol):
            logger.warning(
                f"{img_path.name} differs by more than and atol={atol} from reference."
                "Affine was not snapped, consider re-registering images."
            )
        else:
            if not dry_run:
                img.set_qform(ref_affine)
                img.set_sform(ref_affine)
                nib.save(img, img_path)
            logger.info(f"{img_path.name} affine snapped to NCCT")


def snap_affines(
    data_root: Path,
    modalities: list[str] | None = None,
    reference_mod: str = "ncct",
    atol: float = 1e-4,
    log_file: Path | None = None,
    dry_run: bool = False,
) -> None:
    """Snap image affine transforms if they're closer than a given tolerance.

    Images will be snapped to the NCCT affine. This is necessary to have MONAI
    trasnforms treat each case as a multi-channel image.
    """

    if modalities is None:
        modalities = ["cta", "cbv", "cbf", "mtt", "tmax", "ncct"]

    reference_idx = modalities.index(reference_mod)
    case_dirs = sorted(data_root.glob("train/derivatives/sub-stroke*"))

    with operation_logger("affine_snap_logger", log_file=log_file) as logger:
        logger.info(f"Snapping affines with atol={atol} to reference {reference_mod}")
        for case_dir in case_dirs:
            path_dict = _build_path_dict(case_dir, modalities=modalities)
            channel_list = list(to_paths(path_dict["image"]))
            _snap_affines_image(
                channel_list,
                reference_idx=reference_idx,
                atol=atol,
                logger=logger,
                dry_run=dry_run,
            )
