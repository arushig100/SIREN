"""Load paired images and one count label per scene for the requested question.
Image resizing and normalization follow the selected encoder."""
from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from siren.files import expand_data_path
from siren.settings import ROAD_MASK_COLUMNS

__all__ = [
    "ChipCountDataset",
    "collate_chips",
    "find_unreadable_images",
    "load_split_index",
]


def load_split_index(split_csv: str | Path) -> dict[str, dict]:
    """Map scene IDs to splits, image paths, disaster IDs and whether a road mask is listed."""
    out = {}
    with open(split_csv, newline="") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        missing = {"pre_image_path", "post_image_path", *ROAD_MASK_COLUMNS} - cols
        if missing:
            raise ValueError(f"Split CSV {split_csv} lacks columns {sorted(missing)}. "
                             "Use assets/metadata/split.csv.")
        for r in reader:
            # Expand ${DM3_ROOT} and other environment variables in image paths.
            out[r["chip_id"]] = {
                "split":            r["split"],
                "disaster_id":      r["disaster_id"],
                "pre_image_path":   expand_data_path(r["pre_image_path"]),
                "post_image_path":  expand_data_path(r["post_image_path"]),
                "has_road_mask":    any(r[col] for col in ROAD_MASK_COLUMNS),
            }
    return out


def find_unreadable_images(split_index: dict) -> list[dict]:
    """Return the missing, empty or unreadable pre and post images of every scene."""
    def inspect(item):
        """Check that one image file can be read."""
        cid, field, path = item
        issue = None
        if not path:
            issue = "missing_path"
        elif not os.path.exists(path):
            issue = "missing_file"
        elif os.path.getsize(path) == 0:
            issue = "zero_byte"
        else:
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception as exc:  # PIL raises several format-specific errors
                issue = f"unreadable:{type(exc).__name__}"
        return ({"chip_id": cid, "field": field, "path": path, "issue": issue}
                if issue else None)

    items = [(cid, field, row.get(field)) for cid, row in split_index.items()
             for field in ("pre_image_path", "post_image_path")]
    with ThreadPoolExecutor(max_workers=16) as pool:
        return [issue for issue in pool.map(inspect, items) if issue]


class ChipCountDataset(Dataset):
    """Paired scene images and the scene's reference count.

    Counts come from a reference CSV (chip_id, count); image paths and disaster IDs come from
    the split file."""

    def __init__(self, counts_csv: str | Path, split_index: dict, transform,
                 excluded_chip_ids: set[str] | None = None):
        """Match image pairs to count labels for the selected split."""
        self.tx = transform
        self.rows: list[dict] = []
        with open(counts_csv, newline="") as f:
            for r in csv.DictReader(f):
                cid = r["chip_id"]
                if excluded_chip_ids is not None and cid in excluded_chip_ids:
                    continue
                sp = split_index.get(cid)
                if not sp:
                    raise KeyError(f"Scene {cid!r} in {counts_csv} is missing from the split file")
                pre, post = sp.get("pre_image_path"), sp.get("post_image_path")
                if not pre or not post:
                    raise ValueError(
                        f"Scene {cid!r} has no pre or post image path: pre={pre!r} post={post!r}")
                self.rows.append({
                    "chip_id": cid,
                    "disaster_id": sp.get("disaster_id", ""),
                    "pre": pre, "post": post,
                    "count": float(r["count"]),
                })
        if not self.rows:
            raise ValueError(f"No scenes loaded from {counts_csv}")

    def __len__(self) -> int:
        """Return the number of labelled image pairs."""
        return len(self.rows)

    def _img(self, path: str) -> torch.Tensor:
        """Read an RGB image and apply the encoder preprocessing."""
        return self.tx(Image.open(path).convert("RGB"))

    def __getitem__(self, i: int) -> dict:
        """Return both images, the count label and scene identifiers."""
        r = self.rows[i]
        return {
            "pre": self._img(r["pre"]),
            "post": self._img(r["post"]),
            "count": torch.tensor(r["count"], dtype=torch.float32),
            "chip_id": r["chip_id"],
            "disaster_id": r["disaster_id"],
        }


def collate_chips(batch: list[dict]) -> dict:
    """Stack image tensors and counts, keeping scene IDs for evaluation."""
    return {
        "pre": torch.stack([b["pre"] for b in batch]),
        "post": torch.stack([b["post"] for b in batch]),
        "count": torch.stack([b["count"] for b in batch]),
        "chip_id": [b["chip_id"] for b in batch],
        "disaster_id": [b["disaster_id"] for b in batch],
    }
