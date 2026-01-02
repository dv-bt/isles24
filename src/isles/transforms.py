"""
Code for image processing and transforms
"""

from typing import Sequence, Mapping
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
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    EnsureTyped,
    MapTransform,
    ScaleIntensityRange,
)


def get_train_transforms(
    target_spacing: Sequence[float],
    modalitites: Sequence[str],
    roi_size: Sequence[int] = (64, 64, 64),
    intensity_windows: Mapping[str, Sequence[float]] | None = None,
):
    """
    Build training transforms.

    Parameters
    ----------
    target_spacing : Sequence[float]
        Target voxel spacing in mm (x, y, z).
    modalitites: Sequence[str]
        Order of channel modalities in the ["image"] key.
    roi_size : Sequence[int]
        Size of random crops for training.
    intensity_windows : Mapping[str, Sequence[float]] | None
        Intensity windows for each channel, e.g. {"cta": (a_min, a_max)}. If None,
        no windowing is performed.


    Returns
    -------
    Compose
        MONAI composed transforms.
    """
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=target_spacing,
                mode=("bilinear", "nearest"),
            ),
            PerChannelScaleIntensityd(
                keys=["image"],
                modalities=modalitites,
                windows=intensity_windows,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=roi_size,
                pos=1,
                neg=1,
                num_samples=4,
            ),
            # Data augmentation
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def get_val_transforms(
    target_spacing: Sequence[float],
    modalitites: Sequence[str],
    intensity_windows: Mapping[str, Sequence[float]] | None = None,
):
    """
    Build validation transforms (no augmentation, no random cropping).

    Parameters
    ----------
    target_spacing : tuple of float
        Target voxel spacing in mm.
    modalitites: Sequence[str]
        Order of channel modalities in the ["image"] key.
    intensity_windows : Mapping[str, Sequence[float]] | None
        Intensity windows for each channel, e.g. {"cta": (a_min, a_max)}. If None,
        no windowing is performed.

    Returns
    -------
    Compose
        MONAI composed transforms.
    """
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=target_spacing,
                mode=("bilinear", "nearest"),
            ),
            PerChannelScaleIntensityd(
                keys=["image"],
                modalities=modalitites,
                windows=intensity_windows,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


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

            d[key] = torch.cat(scaled_channels, dim=0)

        return d
