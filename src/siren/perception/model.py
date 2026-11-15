"""Shared SatlasPretrain encoder, building detector, road segmentation head and MLP classifiers.
Damage and destruction use the convolutional heads in siren.inference.buildings."""
from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision.ops.poolers import MultiScaleRoIAlign

from siren.concepts.registry import ConceptSpec, get_concepts
from siren.segmentation.building.model import build_maskrcnn_satlas
from siren.segmentation.common.satlas_backbone import FPN_CHANNELS, SATLAS_FEATURE_CHANNELS
from siren.segmentation.road.model import UPerNetDecoder
from siren.settings import CONTEXT_PADS, DEFAULT_CONTEXT_PAD, IMAGE_SIZE

POOL_SIZE = 7                                       # RoIAlign output grid (7x7)
POOLED_DIM = FPN_CHANNELS * POOL_SIZE * POOL_SIZE   # 6272
CONCEPT_LEVELS = ["0", "1", "2", "3"]    # as in the box RoIAlign; 'pool' is too coarse
# The MLP classifiers built by default and stored in base.pth. The residential and vegetation
# classifiers have their own files (siren.perception.loading.load_checkpoint).
BASE_CLASSIFIERS = ("is_road_blocked",)
# RoIAlign has no weights, so one instance serves every model and device.
_POOL = MultiScaleRoIAlign(featmap_names=CONCEPT_LEVELS, output_size=POOL_SIZE, sampling_ratio=2)


def context_pad(name: str) -> tuple[float, float]:
    """(floor in pixels, fraction of the longer box side) of context padding for a classifier."""
    return CONTEXT_PADS.get(name, DEFAULT_CONTEXT_PAD)


def pool_boxes(fpn, boxes: torch.Tensor, pad: tuple[float, float]) -> torch.Tensor:
    """Pool 7x7 features of boxes in one 1024-pixel image: (N, 6272).

    fpn is that image's feature pyramid. Each box grows on every side by
    max(pad[0], pad[1] x its longer side), taking each side as at least 1 pixel, and is
    clipped to the image."""
    floor, fraction = pad
    if boxes.numel():
        wh = (boxes[:, 2:] - boxes[:, :2]).clamp(min=1)
        grow = (fraction * wh.max(dim=1).values).clamp(min=floor)[:, None]
        boxes = torch.cat(((boxes[:, :2] - grow).clamp(min=0),
                           (boxes[:, 2:] + grow).clamp(max=IMAGE_SIZE)), dim=1)
    levels = OrderedDict((k, fpn[k]) for k in CONCEPT_LEVELS)
    return _POOL(levels, [boxes], [(IMAGE_SIZE, IMAGE_SIZE)]).flatten(1)


def _combine(f_pre: torch.Tensor, f_post: torch.Tensor | None) -> torch.Tensor:
    """Join pre, post and absolute-difference features; a single-image head uses pre only."""
    if f_post is None:
        return f_pre
    return torch.cat([f_pre, f_post, (f_post - f_pre).abs()], dim=-1)


def make_head(in_dim: int, n_classes: int = 2) -> nn.Module:
    """Per-classifier MLP: Linear(in -> 32) -> ReLU -> Dropout(0.2) -> Linear(32 -> classes)."""
    return nn.Sequential(
        nn.Linear(in_dim, 32),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(32, n_classes),
    )


class SharedPerceptionModel(nn.Module):
    """One SatlasPretrain Swin-B encoder shared by the building detector, road segmentation head
    and classifiers.

    concepts lists the MLP classifiers to build (default: BASE_CLASSIFIERS).
    pretrained_backbone=False skips the Satlas weight download when a checkpoint supplies every
    weight."""

    def __init__(self, *, concepts: list[ConceptSpec] | None = None,
                 pretrained_backbone: bool = True):
        """Build the shared encoder, entity extraction models and the given classifiers."""
        super().__init__()
        self.concepts = (list(concepts) if concepts is not None
                         else get_concepts(list(BASE_CLASSIFIERS)))

        # torchvision Mask R-CNN; its backbone is the shared encoder.
        self.det = build_maskrcnn_satlas(image_size=IMAGE_SIZE, pretrained=pretrained_backbone)
        self.backbone = self.det.backbone

        # UPerNet road segmentation head.
        self.road_head = UPerNetDecoder(in_channels=SATLAS_FEATURE_CHANNELS, num_classes=1)

        # Classifiers: a 6272 -> 64 projection of the pooled features and an MLP head for each.
        self.concept_proj = nn.ModuleDict({
            c.name: nn.Sequential(nn.Linear(POOLED_DIM, 64), nn.ReLU(inplace=True), nn.Dropout(0.1))
            for c in self.concepts})
        self.heads = nn.ModuleDict({
            c.name: make_head(in_dim=c.head_input_dim, n_classes=len(c.classes))
            for c in self.concepts})

    def fpn(self, img: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
        """Feature pyramid of an image batch: levels '0' to '3' and 'pool'."""
        return self.backbone(img)

    def fpn_levels(self, fpn_dict) -> list[torch.Tensor]:
        """The four levels read by the road segmentation head and the classifiers."""
        return [fpn_dict[k] for k in CONCEPT_LEVELS]

    @torch.no_grad()
    def verify_backbone_channels(self, device=None, size: int = 256) -> None:
        """Check that every feature-pyramid level has the required 128 channels."""
        dev = device or next(self.parameters()).device
        x = torch.zeros(1, 3, size, size, device=dev)
        d = self.backbone(x)
        for k in CONCEPT_LEVELS:
            c = d[k].shape[1]
            if c != FPN_CHANNELS:
                raise RuntimeError(
                    f"Backbone FPN level '{k}' has {c} channels, expected {FPN_CHANNELS}")

    def pooled(self, fpn_dict, boxes: torch.Tensor, concept_name: str) -> torch.Tensor:
        """Pool and flatten entity features with the classifier's context padding: (N, 6272)."""
        return pool_boxes(fpn_dict, boxes, context_pad(concept_name))

    def project(self, concept_name: str, pooled_feat: torch.Tensor) -> torch.Tensor:
        """Apply the classifier's projection to pooled features, for example cached ones."""
        return self.concept_proj[concept_name](pooled_feat)

    def concept_logits(self, concept_name: str, boxes: torch.Tensor,
                       fpn_pre, fpn_post=None) -> torch.Tensor:
        """Two-class logits per box; paired classifiers also read the post-image features."""
        c = next(x for x in self.concepts if x.name == concept_name)
        f_pre = self.project(concept_name, self.pooled(fpn_pre, boxes, concept_name))
        f_post = (self.project(concept_name, self.pooled(fpn_post, boxes, concept_name))
                  if (c.is_paired and fpn_post is not None) else None)
        return self.heads[concept_name](_combine(f_pre, f_post))

    def road_logits(self, pre: torch.Tensor) -> torch.Tensor:
        """Road logits (B, 1, H, W) from the pre-disaster image at full resolution."""
        feats = self.fpn_levels(self.fpn(pre))
        return self.road_head(feats, (pre.shape[-2], pre.shape[-1]))

    def freeze_backbone(self):
        """Fix the shared encoder (no gradients, evaluation mode) for classifier training."""
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
