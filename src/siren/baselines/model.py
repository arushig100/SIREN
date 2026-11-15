"""Predict a scene count from pre-image, post-image and absolute-difference features."""
from __future__ import annotations

import torch
import torch.nn as nn

from siren.baselines.encoders import Encoder

HEAD_HIDDEN = 256


class ChipCountModel(nn.Module):
    """Share an encoder across both images and predict one question's scene count.

    Each benchmark question has a separately trained instance of this model.
    """

    def __init__(self, encoder: Encoder, dropout: float = 0.1):
        """Attach the image encoder and build a count-prediction head."""
        super().__init__()
        self.encoder = encoder
        d = int(encoder.feat_dim)
        self.head = nn.Sequential(
            nn.Linear(3 * d, HEAD_HIDDEN),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(HEAD_HIDDEN, 1),
        )

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        """Predict a count from pre, post and absolute-difference features."""
        f_pre = self.encoder(pre)
        f_post = self.encoder(post)
        # (B, 3 x feat_dim): pre, post and their absolute difference.
        feat = torch.cat([f_pre, f_post, (f_post - f_pre).abs()], dim=-1)
        return self.head(feat).squeeze(-1)
