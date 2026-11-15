"""RemoteCLIP and SkyCLIP at the paper's 1022-pixel input resolution."""
from __future__ import annotations

import re

import open_clip
import torch

from . import Encoder


class OpenClipEncoder(Encoder):
    """CLIP image encoder; subclasses implement _build(ckpt) -> (clip_model, preprocess)."""

    feat_dim = 768   # ViT-L/14

    def __init__(self, ckpt=None, image_size=1022, **_):
        """Check the input size and load the CLIP encoder."""
        super().__init__()
        if not ckpt:
            raise ValueError("CLIP baselines require their pretrained checkpoint")
        if image_size <= 0 or image_size % 14:
            raise ValueError("CLIP input size must be a positive multiple of 14")
        self.image_size = image_size
        self.clip, self._preprocess = self._build(ckpt)

    def _build(self, ckpt):
        """Create the selected CLIP model."""
        raise NotImplementedError

    def transform(self, image_size):
        """Resize and normalize images for this CLIP checkpoint."""
        if image_size != self.image_size:
            raise ValueError(f"Expected image size {self.image_size}, got {image_size}")
        return self._preprocess

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 768-value CLIP image embedding of each image."""
        return self.clip.encode_image(x).float()


def _remap_timm_to_openclip_visual(sd: dict) -> dict:
    """Rename the keys of SkyCLIP's timm-style visual ViT to open_clip's `visual.*` names."""
    out = {}
    for k, v in sd.items():
        nk = None
        m = re.match(r"blocks\.(\d+)\.(.+)$", k)
        if m:
            blk, rest = m.group(1), m.group(2)
            mp = {
                "norm1.weight": "ln_1.weight", "norm1.bias": "ln_1.bias",
                "norm2.weight": "ln_2.weight", "norm2.bias": "ln_2.bias",
                "attn.qkv.weight": "attn.in_proj_weight", "attn.qkv.bias": "attn.in_proj_bias",
                "attn.proj.weight": "attn.out_proj.weight", "attn.proj.bias": "attn.out_proj.bias",
                "mlp.fc1.weight": "mlp.c_fc.weight", "mlp.fc1.bias": "mlp.c_fc.bias",
                "mlp.fc2.weight": "mlp.c_proj.weight", "mlp.fc2.bias": "mlp.c_proj.bias",
            }
            if rest in mp:
                nk = f"visual.transformer.resblocks.{blk}.{mp[rest]}"
        elif k == "cls_token":
            nk = "visual.class_embedding"
            v = v.reshape(-1)
        elif k == "pos_embed":
            nk = "visual.positional_embedding"
            v = v.reshape(v.shape[1], v.shape[2])
        elif k == "patch_embed.proj.weight":
            nk = "visual.conv1.weight"
        elif k == "patch_embed.proj.bias":
            continue   # open_clip conv1 has no bias
        elif k == "norm_pre.weight":
            nk = "visual.ln_pre.weight"
        elif k == "norm_pre.bias":
            nk = "visual.ln_pre.bias"
        elif k == "norm.weight":
            nk = "visual.ln_post.weight"
        elif k == "norm.bias":
            nk = "visual.ln_post.bias"
        elif k in ("head.weight", "head.bias"):
            continue   # the caller maps head.weight to visual.proj when shapes match
        if nk:
            out[nk] = v
    return out


class RemoteClip(OpenClipEncoder):
    """RemoteCLIP ViT-L/14 image encoder."""

    def _build(self, ckpt):
        """Load the RemoteCLIP checkpoint."""
        S = self.image_size
        # A 1022-pixel input gives a 73 x 73 grid of 14-pixel patches; the checkpoint's
        # position embeddings are interpolated to that grid.
        clip, _, pre = open_clip.create_model_and_transforms("ViT-L-14", force_image_size=S)
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        open_clip.model.resize_pos_embed(sd, clip)
        msg = clip.load_state_dict(sd, strict=False)
        miss = sum(1 for k in msg.missing_keys if k.startswith("visual."))
        if miss > 1:
            raise ValueError(f"{ckpt}: {miss} image-encoder weights missing; expected a RemoteCLIP ViT-L/14 file")
        assert tuple(clip.visual.grid_size) == (S // 14, S // 14), f"grid {clip.visual.grid_size} != {S//14}: force_image_size ignored"
        return clip, pre


class SkyClip(OpenClipEncoder):
    """SkyCLIP ViT-L/14 image encoder."""

    def _build(self, ckpt):
        """Load the SkyCLIP checkpoint."""
        S = self.image_size
        # Start from OpenAI CLIP (downloaded by open_clip); SkyCLIP's image-encoder weights,
        # and its projection when the shapes match, replace them.
        clip, _, pre = open_clip.create_model_and_transforms("ViT-L-14-quickgelu", pretrained="openai", force_image_size=S)
        raw = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        hw = sd.get("head.weight")
        sd = _remap_timm_to_openclip_visual(sd)
        if hw is not None and tuple(hw.shape) == tuple(clip.visual.proj.t().shape):
            sd["visual.proj"] = hw.t().contiguous()
        else:
            print(f"[skyclip] head.weight {None if hw is None else tuple(hw.shape)} not mapped; OpenAI projection kept", flush=True)
        if "visual.positional_embedding" in sd:
            open_clip.model.resize_pos_embed(sd, clip)
        msg = clip.load_state_dict(sd, strict=False)
        miss = sum(1 for k in msg.missing_keys if k.startswith("visual."))
        if miss > 1:
            raise ValueError(f"{ckpt}: {miss} image-encoder weights missing; expected a SkyCLIP ViT-L/14 file in timm format")
        assert tuple(clip.visual.grid_size) == (S // 14, S // 14), f"grid {clip.visual.grid_size} != {S//14}: force_image_size ignored"
        return clip, pre
