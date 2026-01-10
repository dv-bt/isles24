"""
Code for image processing and transforms
"""

from pathlib import Path
from collections.abc import Sequence, Mapping
import numpy as np
from numpy.typing import DTypeLike
import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    CropForegroundd,
    RandFlipd,
    RandRotate90d,
    EnsureTyped,
    MapTransform,
    ScaleIntensityRange,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandCropByLabelClassesd,
    CastToTyped,
    SpatialPadd,
    DeleteItemsd,
    Activationsd,
    AsDiscreted,
    Invertd,
    SaveImaged,
)
from monai.utils import convert_to_dst_type
from monai.data import DataLoader

from isles.swin.config import SwinTrainConfig


def get_train_transforms(config: SwinTrainConfig):
    """
    Build training transforms.

    Parameters
    ----------
    config : SwinTrainConfig
        Configuration dataclass for training multi-encoder Swin-UNETR.

    Returns
    -------
    Compose
        MONAI composed transforms.
    """

    transforms = [
        LoadImaged(keys=["image", "label", "guide"], image_only=False),
        EnsureChannelFirstd(keys=["image", "label", "guide"]),
        Orientationd(keys=["image", "label", "guide"], axcodes="RAS", labels=None),
        CropForegroundd(
            keys=["image", "label"],
            source_key="guide",
            select_fn=lambda x: x > config.fg_threshold,
            margin=10,
        ),
        DeleteItemsd(keys=["guide"]),
        PerChannelScaleIntensityd(
            keys=["image"],
            modalities=config.modalities,
            windows=config.intensity_windows,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
    ]

    if config.target_spacing is not None:
        transforms.append(
            Spacingd(
                keys=["image", "label"],
                pixdim=config.target_spacing,
                mode=("bilinear", "nearest"),
            )
        )

    transforms.extend(
        [
            CastToTyped(keys=["image", "label"], dtype=[torch.float32, torch.uint8]),
            EnsureTyped(keys=["image", "label"], track_meta=True),
            SpatialPadd(keys=["image", "label"], spatial_size=config.roi_size),
            RandCropByLabelClassesd(
                keys=["image", "label"],
                label_key="label",
                num_classes=config.num_classes,
                ratios=config.crop_ratios,
                num_samples=config.num_crops_per_image,
                spatial_size=config.roi_size,
                warn=False,
            ),
            RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.1),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.1),
        ]
    )

    return Compose(transforms)


def get_val_transforms(config: SwinTrainConfig):
    """
    Build validation transforms (no augmentation).

    Parameters
    ----------
    config : SwinTrainConfig
        Configuration dataclass for training multi-encoder Swin-UNETR.

    Returns
    -------
    Compose
        MONAI composed transforms.
    """

    transforms = [
        LoadImaged(keys=["image", "label", "guide"], image_only=False),
        EnsureChannelFirstd(keys=["image", "label", "guide"]),
        Orientationd(keys=["image", "label", "guide"], axcodes="RAS", labels=None),
        CropForegroundd(
            keys=["image", "label"],
            source_key="guide",
            select_fn=lambda x: x > config.fg_threshold,
            margin=10,
        ),
        DeleteItemsd(keys=["guide"]),
        PerChannelScaleIntensityd(
            keys=["image"],
            modalities=config.modalities,
            windows=config.intensity_windows,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
    ]

    if config.target_spacing is not None:
        transforms.append(
            Spacingd(
                keys=["image", "label"],
                pixdim=config.target_spacing,
                mode=("bilinear", "nearest"),
            )
        )

    transforms.extend(
        [
            CastToTyped(keys=["image", "label"], dtype=[torch.float32, torch.uint8]),
            EnsureTyped(keys=["image", "label"], track_meta=True),
        ]
    )

    return Compose(transforms)


class PerChannelScaleIntensityd(MapTransform):
    """
    Rescale images and apply different intensity windowing per channel.

    Parameters
    ----------
    keys : str
        Key for the multichannel image.
    modalitites: Sequence[str]
        Order of channel modalities in the ["image"] key.
    windows : Mapping[str, Sequence[float]] | None
        Intensity windows for each channel, e.g. {"cta": (a_min, a_max)}. If None,
        no windowing is performed, and the whole data range is used instead.
        Default is None.
    b_min : float
        Output minimum.
    b_max : float
        Output maximum.
    clip : bool
        Whether to clip values outside range.
    dtype : DTypeLike
        Output data type, if None, same as input image. defaults to float32.
    """

    def __init__(
        self,
        keys: str,
        modalities: Sequence[str],
        windows: Mapping[str, Sequence[float]] | None = None,
        b_min: float = 0.0,
        b_max: float = 1.0,
        clip: bool = True,
        dtype: DTypeLike = np.float32,
    ):
        super().__init__(keys)
        self.modalities = modalities
        self.windows = windows
        self.b_min = b_min
        self.b_max = b_max
        self.clip = clip
        self.dtype = dtype

    def __call__(self, data):
        d = dict(data)

        for key in self.keys:
            image: np.ndarray = d[key]

            scaled_channels = []
            for c, modality in enumerate(self.modalities):
                channel = image[c : c + 1]

                # Assign windows values
                if self.windows:
                    window = (
                        self.windows[modality]
                        if self.windows[modality]
                        else (None, None)
                    )
                else:
                    window = (None, None)

                a_min = window[0] if window[0] else channel.min()
                a_max = window[1] if window[1] else channel.max()

                scaling = ScaleIntensityRange(
                    a_min=a_min,
                    a_max=a_max,
                    b_min=self.b_min,
                    b_max=self.b_max,
                    clip=self.clip,
                    dtype=self.dtype,
                )

                scaled_channels.append(scaling(channel))

            ret = torch.cat(scaled_channels, dim=0)
            ret = convert_to_dst_type(ret, image, dtype=self.dtype)[0]
            d[key] = ret

        return d


def get_post_transforms(val_loader: DataLoader, out_dir: Path | None = None) -> Compose:
    """
    Build post processing transforms for prediction at original spacing.

    Parameters
    ----------
    val_loader : DataLoader
        Validation dataloader. This is necessary to read the correct transforms to
        invert.
    out_dir : Path | None
        Directory where to save predictions. If None, do not save predictions to disk.
        Default is None

    Returns
    -------
    Compose
        MONAI composed transforms.
    """

    val_transforms = val_loader.dataset.transform

    transforms = [
        Activationsd(keys="pred", softmax=True),
        Invertd(
            keys="pred",
            transform=val_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        AsDiscreted(keys="pred", argmax=True),
    ]

    if out_dir is not None:
        transforms.append(
            SaveImaged(
                keys="pred",
                meta_keys="pred_meta_dict",
                output_dir=out_dir,
                output_postfix="pred",
                resample=False,
                separate_folder=False,
                dtype=np.uint8,
            )
        )

    return Compose(transforms)
