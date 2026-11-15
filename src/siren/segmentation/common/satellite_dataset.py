"""Load paired images with building instance targets, and road masks from the split file.
Instance targets contain boxes, labels and masks rasterized from entity polygons."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from siren.files import expand_data_path, require_file
from siren.settings import ROAD_MASK_COLUMNS


def _mask_paths(row) -> list[str]:
    """Non-empty road-mask paths of a split-file row (a dict or a pandas row)."""
    paths = []
    for column in ROAD_MASK_COLUMNS:
        value = row.get(column)
        if isinstance(value, str) and value:
            paths.append(expand_data_path(value))
    return paths


def has_road_mask(row) -> bool:
    """True when a split-file row lists at least one road mask."""
    return bool(_mask_paths(row))


def read_road_mask(row, image_size: int = 1024) -> np.ndarray:
    """Union of the road masks listed in a split-file row: an (H, W) int64 array of 0 and 1.

    An empty path means no mask; a listed mask that cannot be read raises."""
    target = np.zeros((image_size, image_size), dtype=np.uint8)
    for path in _mask_paths(row):
        try:
            with Image.open(path) as im:
                m = np.asarray(im)
        except OSError as exc:
            raise RuntimeError(f"Cannot read road mask {path}: {exc}. Check that DM3_ROOT points "
                               "to the DisasterM3 folder.") from exc
        if m.ndim == 3:
            m = m[..., 0]
        m = (m > 0).astype(np.uint8)
        if m.shape[0] != image_size or m.shape[1] != image_size:
            m = np.asarray(Image.fromarray(m).resize((image_size, image_size), Image.NEAREST),
                           dtype=np.uint8)
        target = np.maximum(target, m)
    return target.astype(np.int64)


class BuildingInstanceDataset(Dataset):
    """One sample per scene with a reference building file: chip_id, disaster_id, pre and post
    RGB arrays and instance targets (boxes, labels and masks rasterized from the polygons)."""

    def __init__(self, *, split_csv: Path | str, building_entities_dir: Path | str, split: str,
                 image_size: int = 1024) -> None:
        """Select the split's scenes that have a reference building file."""
        self.image_size = int(image_size)
        self.building_entities_dir = Path(building_entities_dir)
        df = pd.read_csv(split_csv)
        df = df[df["split"] == split].reset_index(drop=True)
        keep = [(self.building_entities_dir / f"{chip_id}.json").exists()
                for chip_id in df["chip_id"]]
        self.df = df[keep].reset_index(drop=True)
        if len(self.df) == 0:
            raise RuntimeError(f"No scenes remain for split={split}. Check the entity folder "
                               f"{building_entities_dir}.")

    def __len__(self) -> int:
        """Return the number of selected scenes."""
        return len(self.df)

    def _read_image(self, path: str) -> np.ndarray:
        """Read an RGB image as (H, W, 3) uint8; an unreadable file raises."""
        with Image.open(require_file(expand_data_path(path))) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        if arr.shape[0] != self.image_size or arr.shape[1] != self.image_size:
            # Resize to image_size; DisasterM3 images are already 1024 x 1024.
            im = Image.fromarray(arr).resize((self.image_size, self.image_size), Image.BILINEAR)
            arr = np.asarray(im, dtype=np.uint8)
        return arr

    def _rasterize_polygon(self, poly: list[list[float]], size: int) -> np.ndarray:
        """Rasterize a polygon (list of [x, y]) to a (size, size) binary mask."""
        img = Image.new("L", (size, size), 0)
        if len(poly) >= 3:
            draw = ImageDraw.Draw(img)
            pts = [(float(x), float(y)) for x, y in poly]
            draw.polygon(pts, fill=1)
        return np.asarray(img, dtype=np.uint8)

    def _build_instance_target(self, record: dict[str, Any]) -> dict[str, Any]:
        """Build the instance-segmentation target from a building entity JSON."""
        entities = (record.get("entities") or {}).get("buildings", [])
        boxes = []
        masks = []
        labels = []
        for e in entities:
            geom = e.get("geometry") or {}
            bbox = geom.get("bbox_xyxy")
            polys = geom.get("polygon_xy") or []
            if not bbox or len(bbox) != 4 or not polys:
                continue
            # polygon_xy is a list of rings; rasterize only the ring with the most points.
            mask = self._rasterize_polygon(max(polys, key=len), self.image_size)
            if mask.sum() <= 0:
                continue
            boxes.append([float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])])
            masks.append(mask)
            labels.append(0)  # single foreground class
        if not boxes:
            return {
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "masks": np.zeros((0, self.image_size, self.image_size), dtype=np.uint8),
                "labels": np.zeros((0,), dtype=np.int64),
            }
        return {
            "boxes": np.array(boxes, dtype=np.float32),
            "masks": np.stack(masks, axis=0),
            "labels": np.array(labels, dtype=np.int64),
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Load the requested scene; raise RuntimeError if an input cannot be read."""
        row = self.df.iloc[idx]
        chip_id = row["chip_id"]
        try:
            record = json.loads((self.building_entities_dir / f"{chip_id}.json").read_text())
            pre = self._read_image(row["pre_image_path"])
            post = self._read_image(row["post_image_path"])
        except Exception as exc:
            raise RuntimeError(f"Cannot load scene {chip_id}: {exc}") from exc
        return {"chip_id": chip_id, "disaster_id": row["disaster_id"],
                "pre": pre, "post": post, "target": self._build_instance_target(record)}


def to_tensor_chw(arr: np.ndarray) -> torch.Tensor:
    """(H, W, 3) uint8 -> (3, H, W) float32 in [0, 1]."""
    t = torch.from_numpy(np.ascontiguousarray(arr).copy()).float() / 255.0
    if t.ndim == 3 and t.shape[-1] == 3:
        t = t.permute(2, 0, 1)
    return t


def collate_instance(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack the images and keep the instance targets as a list of dicts."""
    pre = torch.stack([to_tensor_chw(b["pre"]) for b in batch], dim=0)
    post = torch.stack([to_tensor_chw(b["post"]) for b in batch], dim=0)
    targets = []
    for b in batch:
        t = b["target"]
        targets.append({
            "boxes": torch.from_numpy(t["boxes"]),
            "masks": torch.from_numpy(t["masks"]),
            "labels": torch.from_numpy(t["labels"]),
        })
    return {
        "pre": pre,
        "post": post,
        "targets": targets,
        "chip_ids": [b["chip_id"] for b in batch],
        "disaster_ids": [b["disaster_id"] for b in batch],
    }

