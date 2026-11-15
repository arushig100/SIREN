"""Encode images with Satlas Aerial Swin-B and its FPN, averaging the input-resolution
feature map. Inputs are in [0, 1]."""
from __future__ import annotations

import satlaspretrain_models
import torch
import torch.nn as nn
from torchvision import transforms

from . import Encoder

MODEL_ID = "Aerial_SwinB_SI"


class SatlasEncoder(Encoder):
    """Satlas Aerial Swin-B encoder with its FPN."""

    feat_dim = 128   # replaced in __init__ by the measured channel count

    def __init__(self, ckpt: str | None = None, **_):
        """Load the Satlas aerial-image encoder; ckpt is unused (the library downloads the weights)."""
        super().__init__()
        self.satlas = satlaspretrain_models.Weights().get_pretrained_model(
            model_identifier=MODEL_ID, fpn=True, device="cpu")
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # The random input also advances the seeded random stream that training continues from.
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            self.feat_dim = int(self.satlas(dummy)[0].shape[1])

    def transform(self, image_size: int):
        """Resize the image and convert it to [0, 1]; Satlas needs no further normalization."""
        return transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Average the input-resolution Satlas feature map into one vector per image: (B, 128)."""
        return self.pool(self.satlas(x)[0]).flatten(1).float()
