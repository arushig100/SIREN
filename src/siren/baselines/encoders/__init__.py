"""Encoder interface and the four remote-sensing baseline encoders."""
from __future__ import annotations

import importlib
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

__all__ = ["ENCODERS", "Encoder", "build_encoder"]


class Encoder(nn.Module, ABC):
    """A pretrained image encoder. `forward(x:[B,3,H,W]) -> [B, feat_dim]`."""

    feat_dim: int = 0
    name: str = ""

    @abstractmethod
    def transform(self, image_size: int):
        """Return the encoder's own preprocessing transform; the trainer adds paired flips."""
        ...

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode an image batch as one feature vector per image."""
        ...


# Command-line name -> (adapter module in this package, encoder class). The adapter and its
# libraries are imported only when the encoder is built.
ENCODERS = {
    "dinov3": ("dinov3", "DinoV3"),
    "remoteclip": ("clip", "RemoteClip"),
    "satlas": ("satlas", "SatlasEncoder"),
    "skyclip": ("clip", "SkyClip"),
}


def build_encoder(name: str, **kwargs) -> Encoder:
    """Build one of the baseline encoders by its command-line name."""
    if name not in ENCODERS:
        raise KeyError(f"Unknown encoder {name!r}; choose from {sorted(ENCODERS)}")
    module, cls = ENCODERS[name]
    encoder = getattr(importlib.import_module(f"{__name__}.{module}"), cls)(**kwargs)
    encoder.name = name
    return encoder
