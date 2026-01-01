"""
Code for multi encoder Swin-UNETR
"""

import copy
from dataclasses import dataclass
import torch
import torch.nn as nn
from monai.networks.nets.swin_unetr import (
    SwinUNETR,
    filter_swinunetr,
)

from monai.networks.blocks import (
    UnetrBasicBlock,
)
from monai.networks.utils import copy_model_state


@dataclass
class Config:
    max_epochs: int
    use_mixed_precision: bool = True


class MultiEncoderSwinUNETR(SwinUNETR):
    """Swin-UNETR with multi encoders.

    Based on <https://arxiv.org/abs/2201.01266>", inherits all the decoder architecture
    from monai.networks.nets.SwinUNETR and replaces the single encoder with per-modality
    encoder and channel fusion.

    Parameters
    ----------
    modalities : List[str]
        List of modality names (e.g., ["CTA", "CBF"]).
    feature_size : int
        Base feature dimension.
    fusion_kernel_size : int
        Kernel size for fusion convolutions.
    **kwargs
        Additional arguments passed to parent SwinUNETR.
    """

    def __init__(
        self,
        modalities: list[str],
        feature_size: int = 48,
        fusion_kernel_size: int = 1,
        **kwargs,
    ) -> None:

        super().__init__(
            in_channels=1,
            feature_size=feature_size,
            **kwargs,
        )

        self.modalities = modalities
        num_modalities = len(modalities)

        self.swin_encoders = nn.ModuleDict({
            modality: copy.deepcopy(self.swinViT)
            for modality in modalities
        })
        del self.swinViT

        self.fusion_layers = nn.ModuleList([
            nn.Conv3d(
                in_channels=feature_size * mult * num_modalities,
                out_channels=feature_size * mult,
                kernel_size=fusion_kernel_size,
                padding=fusion_kernel_size // 2,
            )
            for mult in [1, 2, 4, 8, 16]
        ])

        # Replace encoder1 to handle multi-channel input
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=kwargs.get("spatial_dims", 3),
            in_channels=num_modalities,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=kwargs.get("norm_name", "instance"),
            res_block=True,
        )

    def load_pretrained_encoders(self, weights_path: str) -> None:
        """Load SSL pretrained weights into all encoders."""
        ssl_weights = torch.load(weights_path, weights_only=False)["state_dict"]
        
        for modality, encoder in self.swin_encoders.items():
            wrapper = nn.Module()
            wrapper.swinViT = encoder
            
            _, loaded, _ = copy_model_state(
                wrapper, ssl_weights, filter_func=filter_swinunetr
            )
            print(f"Encoder [{modality}]: loaded {len(loaded)} keys")

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        assert x_in.shape[1] == len(self.modalities)
        
        # Run each modality through its encoder
        all_hidden_states = [
            self.swin_encoders[modality](x_in[:, i:i+1, ...], self.normalize)
            for i, modality in enumerate(self.modalities)
        ]
        
        # Fuse at each scale
        fused_hidden_states = [
            self.fusion_layers[s](torch.cat([hs[s] for hs in all_hidden_states], dim=1))
            for s in range(5)
        ]
        
        # Decoder (reusing parent's blocks)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(fused_hidden_states[0])
        enc2 = self.encoder3(fused_hidden_states[1])
        enc3 = self.encoder4(fused_hidden_states[2])
        dec4 = self.encoder10(fused_hidden_states[4])
        
        dec3 = self.decoder5(dec4, fused_hidden_states[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        
        logits = self.out(out)
        return logits
