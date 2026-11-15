"""Road segmentation loss with binary cross-entropy, Dice and centerline Dice."""

from __future__ import annotations

import torch
import torch.nn.functional as F


# Soft morphology for the centerline Dice loss.


def _soft_erode(x: torch.Tensor) -> torch.Tensor:
    """Approximate mask erosion with differentiable pooling."""
    p1 = F.max_pool2d(-x, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = F.max_pool2d(-x, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return -torch.max(p1, p2)


def _soft_dilate(x: torch.Tensor) -> torch.Tensor:
    """Approximate mask dilation with differentiable pooling."""
    return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)


def _soft_open(x: torch.Tensor) -> torch.Tensor:
    """Remove small mask structures by erosion followed by dilation."""
    return _soft_dilate(_soft_erode(x))


def soft_skeletonize(probs: torch.Tensor, iters: int = 3) -> torch.Tensor:
    """Approximate a centerline by repeated soft erosion and opening."""
    x = probs
    skel = torch.relu(x - _soft_open(x))
    for _ in range(iters - 1):
        x = _soft_erode(x)
        skel = skel + torch.relu(x - _soft_open(x))
    return torch.clamp(skel, 0.0, 1.0)


def soft_cldice_loss(probs: torch.Tensor, targets: torch.Tensor,
                     iters: int = 3, eps: float = 1.0) -> torch.Tensor:
    """Soft centerline Dice loss (cl-Dice), symmetric in prediction and target."""
    pred_sk = soft_skeletonize(probs, iters=iters)
    tgt_sk = soft_skeletonize(targets, iters=iters)
    tprec = (pred_sk * targets).sum() / (pred_sk.sum() + eps)
    tsens = (tgt_sk * probs).sum() / (tgt_sk.sum() + eps)
    # eps also smooths the harmonic mean, so a perfect prediction scores about 2/3
    # (loss about 1/3).
    cl_dice = 2 * tprec * tsens / (tprec + tsens + eps)
    return 1.0 - cl_dice


def roads_loss(logits: torch.Tensor, gt: torch.Tensor, *,
               dice_weight: float = 4.0,
               cldice_weight: float = 2.0,
               cldice_iters: int = 5,
               pos_weight_max: float = 50.0) -> torch.Tensor:
    """Positive-weighted BCE + dice_weight x Dice + cldice_weight x centerline Dice, in float32 (autocast off).

    logits: (B, 1, H, W) raw; gt: (B, H, W) with values 0 or 1."""
    with torch.amp.autocast(device_type="cuda", enabled=False):
        logits = logits.float()
        gt = gt.float()
        if logits.dim() == 4 and logits.shape[1] == 1:
            logits = logits[:, 0]   # (B, H, W)
        pred = torch.sigmoid(logits).clamp(1e-7, 1.0 - 1e-7)
        # Class-balanced BCE: positive weight = negative / positive pixels, clipped to [1, pos_weight_max].
        n_pos = gt.sum().clamp_min(1.0)
        n_neg = (gt.numel() - n_pos).clamp_min(1.0)
        pos_weight = (n_neg / n_pos).clamp(1.0, pos_weight_max)
        bce = -(pos_weight * gt * torch.log(pred)
                + (1.0 - gt) * torch.log(1.0 - pred)).mean()
        # Dice loss per image, averaged over the batch.
        inter = (pred * gt).sum(dim=(1, 2))
        union = pred.sum(dim=(1, 2)) + gt.sum(dim=(1, 2))
        dice = (1.0 - (2.0 * inter + 1.0) / (union + 1.0)).mean()
        # Centerline Dice rewards connected thin roads.
        cldice = torch.zeros((), device=logits.device)
        if cldice_weight > 0.0:
            cldice = soft_cldice_loss(pred, gt, iters=cldice_iters)
        loss = bce + dice_weight * dice + cldice_weight * cldice
        if not torch.isfinite(loss):
            # Stop rather than write NaN into the weights.
            raise FloatingPointError('Road loss is not finite; the road logits contain NaN or Inf')
    return loss
