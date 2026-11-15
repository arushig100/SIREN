"""Mask R-CNN building detection using a Satlas Swin-B encoder."""
from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn

from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops.poolers import MultiScaleRoIAlign

from siren.segmentation.common.satlas_backbone import FPN_CHANNELS, SatlasBackbone


class SatlasMaskRCNNBackbone(nn.Module):
    """Format Satlas features for Mask R-CNN, with an added stride-64 feature level."""

    def __init__(self, *, pretrained: bool = True) -> None:
        """Adapt the Satlas feature pyramid to the building detector."""
        super().__init__()
        self.satlas = SatlasBackbone(pretrained=pretrained)
        # torchvision MaskRCNN reads this attribute to size its heads.
        self.out_channels = FPN_CHANNELS

    def forward(self, x: torch.Tensor) -> OrderedDict:
        """x: (B, 3, H, W) in [0, 1]. Return levels '0' to '3' and the extra 'pool' level."""
        if x.shape[1] != 3:
            raise RuntimeError(f"Expected 3-channel input, got {x.shape[1]} channels")
        feats = self.satlas(x)
        if len(feats) != 4:
            raise RuntimeError(f"Expected 4 feature levels, got {len(feats)}")

        # torchvision expects string level names.
        named = OrderedDict(zip(["0", "1", "2", "3"], feats))
        # Stride-64 'pool' level: kernel=1, stride=2 matches torchvision's
        # LastLevelMaxPool (equivalent to stride-2 subsampling of the stride-32 map).
        pool = torch.nn.functional.max_pool2d(named["3"], kernel_size=1, stride=2)
        named["pool"] = pool
        return named


def build_maskrcnn_satlas(
    *,
    num_classes: int = 2,         # background (0) + building (1)
    image_size: int = 1024,
    pretrained: bool = True,
    detections_per_img: int = 500,
    score_thresh: float = 0.05,
    nms_thresh: float = 0.5,
    anchor_sizes: tuple[tuple[int, ...], ...] = (
        (16,), (32,), (64,), (128,), (256,)
    ),
    aspect_ratios: tuple[float, ...] = (0.5, 1.0, 2.0),
    rpn_pre_nms_top_n_train: int = 2000,
    rpn_post_nms_top_n_train: int = 1000,
    rpn_pre_nms_top_n_test: int = 1000,
    rpn_post_nms_top_n_test: int = 1000,
) -> MaskRCNN:
    """Build Mask R-CNN with Satlas features and one anchor group per pyramid level.

    image_size fixes the input dimensions. num_classes includes background.
    pretrained=False skips the Satlas weight download for callers that load a checkpoint."""
    if len(anchor_sizes) != 5:
        raise ValueError(
            f"anchor_sizes must have 5 entries (one per FPN level), "
            f"got {len(anchor_sizes)}"
        )
    backbone = SatlasMaskRCNNBackbone(pretrained=pretrained)

    # AnchorGenerator wants one (sizes, ratios) pair per FPN level.
    aspect_ratios_per_level = (aspect_ratios,) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes, aspect_ratios=aspect_ratios_per_level
    )

    box_roi_pool = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"],   # exclude "pool": too coarse for boxes
        output_size=7, sampling_ratio=2,
    )
    mask_roi_pool = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"],
        output_size=14, sampling_ratio=2,
    )

    # Satlas takes [0, 1] inputs without further normalization; mean 0 and std 1 make the
    # detector transform's normalization a no-op.
    image_mean = [0.0] * 3
    image_std = [1.0] * 3

    model = MaskRCNN(
        backbone=backbone,
        num_classes=num_classes,
        # min_size == max_size == image_size disables the transform's resize.
        min_size=image_size, max_size=image_size,
        image_mean=image_mean, image_std=image_std,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=box_roi_pool,
        mask_roi_pool=mask_roi_pool,
        box_score_thresh=score_thresh,
        box_nms_thresh=nms_thresh,
        box_detections_per_img=detections_per_img,
        rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train,
        rpn_post_nms_top_n_train=rpn_post_nms_top_n_train,
        rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
        rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
    )
    return model


def maskrcnn_targets(targets: list[dict], device: str) -> list[dict]:
    """Convert dataset targets to Mask R-CNN tensors.

    Shift foreground labels from 0 to 1 because Mask R-CNN reserves 0 for
    background. Keep boxes and masks unchanged; training filters empty targets."""
    out = []
    for t in targets:
        boxes = t["boxes"].to(device).float()
        labels = t["labels"].to(device).long() + 1
        masks = t["masks"].to(device).to(torch.uint8)
        out.append({"boxes": boxes, "labels": labels, "masks": masks})
    return out
