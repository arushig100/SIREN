"""Load the Satlas Aerial Swin-B encoder and return four feature-map scales.
The building detector, road segmentation head and entity classifiers share these features."""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import satlaspretrain_models
except ImportError as exc:
    raise ImportError(
        "satlaspretrain_models is not installed. Install the packages in requirements-model.txt."
    ) from exc


MODEL_ID = "Aerial_SwinB_SI"
# Satlas Swin-B + FPN emits 128 channels at each of the strides 4, 8, 16 and 32.
FPN_CHANNELS = 128
SATLAS_FEATURE_CHANNELS = (FPN_CHANNELS,) * 4


class SatlasBackbone(nn.Module):
    """Return Satlas feature maps at strides 4, 8, 16 and 32, each with 128 channels.

    pretrained=False builds the same architecture without downloading the Satlas
    weights; the caller must then load every tensor from a checkpoint."""

    def __init__(self, pretrained: bool = True) -> None:
        """Load Satlas and expose its multiscale image features."""
        super().__init__()
        if pretrained:
            self.satlas_model = satlaspretrain_models.Weights().get_pretrained_model(
                model_identifier=MODEL_ID, fpn=True, device="cpu")
        else:
            # The same constructor call as get_pretrained_model, without loading weights.
            from satlaspretrain_models.model import Model, SatlasPretrain_weights
            info = SatlasPretrain_weights[MODEL_ID]
            self.satlas_model = Model(info["num_channels"], info["multi_image"], info["backbone"],
                                      fpn=True, head=None, num_categories=None, weights=None)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """x: (B, 3, H, W) in [0, 1]. Returns 4 maps, each (B, 128, H/stride, W/stride).

        satlaspretrain_models 0.3.1 returns five levels; the first, an upsampled stride-1
        level, is dropped."""
        out = self.satlas_model(x)
        if len(out) != 5:
            raise RuntimeError(f"Satlas backbone returned {len(out)} levels, expected 5")
        return list(out[1:])
