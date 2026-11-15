"""DINOv3 SAT-493M ViT-L/16: CLS token and mean patch features."""
from __future__ import annotations

import torch
from torchvision import transforms

from . import Encoder


class DinoV3(Encoder):
    """DINOv3 ViT-L/16 pretrained on SAT-493M satellite images."""

    MODEL = "vit_large_patch16_dinov3.sat493m"

    def __init__(self, ckpt=None, **_):
        """Load the satellite-pretrained DINOv3 encoder."""
        super().__init__()
        import timm
        # Pretrained weights through timm, or a local checkpoint given with --ckpt.
        self.vit = timm.create_model(self.MODEL, pretrained=ckpt is None, num_classes=0,
                                     dynamic_img_size=True, checkpoint_path=ckpt or "")
        cfg = timm.data.resolve_model_data_config(self.vit)
        self._mean, self._std = tuple(cfg["mean"]), tuple(cfg["std"])
        self._prefix = int(getattr(self.vit, "num_prefix_tokens", 1))
        self.feat_dim = 2 * int(self.vit.num_features)  # CLS token and mean patch token

    def transform(self, image_size):
        """Resize to the requested image size and apply checkpoint normalization."""
        if image_size % 16:
            raise ValueError("DINOv3 ViT-L/16 needs an input size divisible by 16")
        return transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(),
                                   transforms.Normalize(self._mean, self._std)])

    def forward(self, x):
        """Concatenate the CLS token and mean spatial-patch features."""
        tokens = self.vit.forward_features(x)                      # [B, prefix + N, C]
        # Average only spatial patch tokens; exclude the CLS and register tokens.
        return torch.cat([tokens[:, 0], tokens[:, self._prefix:].mean(1)], dim=1).float()
