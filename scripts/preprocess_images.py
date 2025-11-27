"""
Preprocess images according to https://doi.org/10.48550/arXiv.2505.18424 and
https://github.com/KurtLabUW/ISLES2024/
"""

import json
import re
from pathlib import Path
from tqdm import tqdm
import numpy as np
import nibabel as nib


def clip_image(image: nib.Nifti1Image, modality: str) -> nib.Nifti1Image:
    """Clip image intensity according to the defined intensity ranges.

    Parameters
    ----------
    image : nib.Nifti1Image
        Image to be clipped
    modality : str
        Modality of the image. It will be used to select the approriate
        windowing range. Accepted values: 'cta', 'cbf', 'cbv', 'mtt', 'tmax'.

    """

    # Custom windowing ranges for each modality
    intensity_ranges = {
        "cta": (0, 90),
        "cbf": (0, 35),
        "cbv": (0, 10),
        "mtt": (0, 20),
        "tmax": (0, 7),
    }

    data = image.get_fdata()
    data_clipped = np.clip(data, *intensity_ranges[modality])
    image_clipped = nib.Nifti1Image(data_clipped, image.affine, image.header)
    return image_clipped


def process_case(case: dict, src_dir: Path, dst_dir: Path) -> list[str]:
    """Process a single case.

    Reads the images in a case, clips them, and saves them under dst_dir while
    preserving the original directory structure relatvie to src_dir.

    Parameters
    ----------
    case : dict
        A dictionary containing the info for a single case in a MONAI datalist
    src_dir : Path
        Path to the source directory of the images. It will be considered as the
        root for preserving the directory structures of each case.
    dst_dir : Path
        Path to the new root directory.

    Returns
    -------
    list[str]
        A list of strings containing the paths to the clipped images, which can
        be used to update the case entry

    """

    new_images = []
    for image_path in case["image"]:
        # Get destination path and image modality
        image_path = Path(image_path)
        dst_path = dst_dir / image_path.relative_to(src_dir)
        dst_path.parent.mkdir(exist_ok=True, parents=True)
        modality = re.search(r"_(\w+).nii.gz$", image_path.name).group(1)

        # Clip and save image
        image = nib.load(image_path)
        image_clipped = clip_image(image, modality=modality)
        nib.save(image_clipped, dst_path)

        new_images.append(str(dst_path))

    return new_images


def main():
    """Read images from a datalist, run preprocessing, and save updated datalist"""

    work_dir = Path("/home/renku/work/data-local")
    datalist_path = work_dir / "processed/datalist-auto3dseg.json"

    with datalist_path.open() as f:
        datalist = json.load(f)

    src_dir = work_dir / "train/derivatives"
    dst_dir = work_dir / "train/preprocessed"
    dst_dir.mkdir(exist_ok=True, parents=True)

    for split in ["training", "testing"]:
        case_list = datalist[split]
        for case in tqdm(case_list, "Processing cases"):
            new_images = process_case(case, src_dir, dst_dir)
            case["image"] = new_images

    new_datalist_path = datalist_path.parent / "datalist-preprocessed.json"
    with new_datalist_path.open(mode="w") as f:
        json.dump(datalist, f, indent=4)


if __name__ == "__main__":
    main()
