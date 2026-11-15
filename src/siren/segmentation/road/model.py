"""UPerNet road segmentation using the four Satlas feature-pyramid levels."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from siren.segmentation.common.satlas_backbone import SATLAS_FEATURE_CHANNELS


def _norm2d(channels: int) -> nn.Module:
    """Group normalization with 32 groups (one group if channels is not a multiple of 32)."""
    return nn.GroupNorm(32 if channels % 32 == 0 else 1, channels)


class PPM(nn.Module):
    """UPerNet pyramid pooling module: pool at several scales and merge with the input."""

    def __init__(self, in_ch: int, out_ch: int,
                 pool_scales: tuple[int, ...] = (1, 2, 3, 6)) -> None:
        """Build pooling branches that collect context at several spatial scales."""
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                _norm2d(out_ch),
                nn.ReLU(inplace=True),
            )
            for s in pool_scales
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_ch + len(pool_scales) * out_ch, out_ch,
                      kernel_size=3, padding=1, bias=False),
            _norm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool and resize context features, then combine them with the input."""
        H, W = x.shape[-2:]
        outs = [x]
        for b in self.branches:
            o = b(x)
            o = F.interpolate(o, size=(H, W), mode="bilinear",
                              align_corners=False)
            outs.append(o)
        return self.bottleneck(torch.cat(outs, dim=1))


class UPerNetDecoder(nn.Module):
    """Pool context at the coarsest scale, merge pyramid levels and predict roads."""

    def __init__(
        self,
        *,
        in_channels: tuple[int, ...] = SATLAS_FEATURE_CHANNELS,
        fpn_channels: int = 256,
        num_classes: int = 1,
        ppm_pool_scales: tuple[int, ...] = (1, 2, 3, 6),
        dropout: float = 0.1,
    ) -> None:
        """Build the multiscale road-segmentation decoder."""
        super().__init__()
        self.num_levels = len(in_channels)

        # Pyramid pooling on the coarsest level.
        top_in = in_channels[-1]
        self.ppm = PPM(top_in, fpn_channels, ppm_pool_scales)

        # 1x1 lateral convolutions for the finer levels.
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, fpn_channels, kernel_size=1, bias=False),
                _norm2d(fpn_channels),
                nn.ReLU(inplace=True),
            )
            for c in in_channels[:-1]
        ])

        # 3x3 smoothing convolutions after each top-down merge.
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3,
                          padding=1, bias=False),
                _norm2d(fpn_channels),
                nn.ReLU(inplace=True),
            )
            for _ in in_channels[:-1]
        ])

        # Concatenated levels, then a 3x3 convolution and a 1x1 classifier.
        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.num_levels * fpn_channels, fpn_channels,
                      kernel_size=3, padding=1, bias=False),
            _norm2d(fpn_channels),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Conv2d(fpn_channels, num_classes, kernel_size=1)
        # Small initial weights avoid large first-step logits under the positive-weighted BCE.
        nn.init.normal_(self.classifier.weight, std=0.01)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, feats: list[torch.Tensor],
                output_size: tuple[int, int]) -> torch.Tensor:
        """Road logits of size output_size (H, W) from feature levels in increasing stride order
        (4, 8, 16 and 32 for Satlas)."""
        assert len(feats) == self.num_levels, \
            f"Expected {self.num_levels} feats, got {len(feats)}"
        assert feats[0].shape[-1] > feats[-1].shape[-1], \
            f"feats must be in increasing-stride order; got shapes {[f.shape for f in feats]}"

        laterals = [conv(f) for conv, f in zip(self.lateral_convs, feats[:-1])]
        top = self.ppm(feats[-1])

        # Top-down merge, starting from the pooled coarsest level.
        merged = [None] * self.num_levels
        merged[-1] = top
        for i in range(self.num_levels - 2, -1, -1):
            up = F.interpolate(merged[i + 1], size=laterals[i].shape[-2:],
                               mode="bilinear", align_corners=False)
            merged[i] = laterals[i] + up

        # Smooth lateral levels only; the PPM top is left as-is.
        for i in range(self.num_levels - 1):
            merged[i] = self.fpn_convs[i](merged[i])

        target_size = merged[0].shape[-2:]
        outs = [merged[0]]
        for i in range(1, self.num_levels):
            outs.append(
                F.interpolate(merged[i], size=target_size, mode="bilinear",
                              align_corners=False)
            )
        x = torch.cat(outs, dim=1)
        x = self.bottleneck(x)
        x = self.dropout(x)
        x = self.classifier(x)

        x = F.interpolate(x, size=output_size, mode="bilinear",
                          align_corners=False)
        return x
