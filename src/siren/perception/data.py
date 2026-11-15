"""Load paired imagery and training labels. Stage 1 uses building instances and road
masks; Stage 2 joins entity boxes to classifier labels by scene and entity ID."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from siren.concepts.registry import ConceptSpec
from siren.files import expand_data_path, read_unreadable_scenes, require_file
from siren.settings import IMAGE_SIZE
from siren.segmentation.common.satellite_dataset import (
    BuildingInstanceDataset, collate_instance, has_road_mask, read_road_mask,
)


def load_image_chw(path: str, image_size: int = IMAGE_SIZE, device="cpu") -> torch.Tensor:
    """Read an RGB image as a (1, 3, H, W) float tensor in [0, 1]."""
    im = Image.open(require_file(path)).convert("RGB")
    if im.size != (image_size, image_size):
        im = im.resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.uint8)
    t = torch.from_numpy(arr).float().div_(255.0).permute(2, 0, 1).contiguous()
    return t.unsqueeze(0).to(device)


# Stage 1: building instances and road masks.
class MultiTaskChipDataset(Dataset):
    """Per scene: pre and post images, building instance targets and a binary road mask.

    Scenes are those with a building entity file. The road mask comes from the split file when
    the scene lists one; scenes with road masks only are not used."""

    def __init__(self, *, split_csv: str | Path, building_entities_dir: str | Path,
                 split: str, image_size: int = IMAGE_SIZE, unreadable_scenes=None):
        """Select the split's scenes with building files; skip unreadable validation scenes."""
        self.image_size = image_size
        self.buildings = BuildingInstanceDataset(
            split_csv=split_csv, building_entities_dir=building_entities_dir,
            split=split, image_size=image_size)
        with open(split_csv, newline="") as f:
            self.road_rows = {r["chip_id"]: r for r in csv.DictReader(f)
                              if r["split"] == split and has_road_mask(r)}
        if unreadable_scenes is not None and split == "val":
            excluded = read_unreadable_scenes(unreadable_scenes)
            df = self.buildings.df
            self.buildings.df = df[~df["chip_id"].isin(excluded)].reset_index(drop=True)

    def sample_scenes(self, n: int, seed: int) -> None:
        """Keep a random sample of n scenes (pandas sampling with random_state=seed)."""
        if n < len(self.buildings.df):
            self.buildings.df = self.buildings.df.sample(
                n=n, random_state=seed).reset_index(drop=True)

    def __len__(self) -> int:
        """Return the number of scenes."""
        return len(self.buildings)

    def __getitem__(self, idx: int) -> dict:
        """Load a scene and its available building and road targets."""
        item = self.buildings[idx]                  # chip_id, disaster_id, pre, post, target
        row = self.road_rows.get(item["chip_id"])
        if row is not None:
            item["road_semantic"] = read_road_mask(row, self.image_size)
            item["road_valid"] = True
        else:  # no road mask: an empty target that the road loss skips
            item["road_semantic"] = np.zeros((self.image_size, self.image_size), dtype=np.int64)
            item["road_valid"] = False
        return item


def collate_multitask(batch: list[dict]) -> dict:
    """Collate images and building targets (collate_instance), road masks and road-label flags."""
    base = collate_instance(batch)                  # pre, post, targets, chip_ids, disaster_ids
    base["road_semantic"] = torch.stack(
        [torch.from_numpy(b["road_semantic"]) for b in batch], dim=0)   # (B,H,W) int64
    base["road_valid"] = torch.tensor([bool(b["road_valid"]) for b in batch], dtype=torch.bool)
    return base


# Stage 2: per-entity labels with reference boxes.
@dataclass(frozen=True)
class ConceptRow:
    """One labelled entity with its reference box and image paths."""
    chip_id: str
    entity_id: str
    bbox: tuple[float, float, float, float]     # x1, y1, x2, y2 in pixels
    label: int                                  # 1 = positive
    pre_image_path: str
    post_image_path: str | None


def _load_split_index(split_csv: str | Path) -> dict[str, dict]:
    """chip_id -> {pre_image_path, post_image_path}."""
    out: dict[str, dict] = {}
    with open(split_csv, newline="") as f:
        for r in csv.DictReader(f):
            out[r["chip_id"]] = {
                # Expand ${DM3_ROOT} and other environment variables in image paths.
                "pre_image_path": expand_data_path(r.get("pre_image_path") or ""),
                "post_image_path": expand_data_path(r.get("post_image_path") or ""),
            }
    return out


def load_reference_boxes(path: str | Path) -> dict[tuple[str, str], list[float]]:
    """Read reference boxes (x1, y1, x2, y2 in pixels) keyed by scene and entity IDs."""
    out: dict[tuple[str, str], list[float]] = {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Box file not found: {p}")
    with p.open(newline="") as f:
        for r in csv.DictReader(f):
            bbox = json.loads(r["bbox"])
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError(f"{p}: bad box for {r['chip_id']} {r['entity_id']}")
            out[(r["chip_id"], r["entity_id"])] = [float(v) for v in bbox]
    return out


def _label_stem(concept_name: str) -> str:
    """Label file stem of a classifier, for example 'is_road_blocked' -> 'road_blocked'."""
    m = {
        "is_damaged": "damaged",
        "is_destroyed": "destroyed",
        "is_residential": "residential",
        "is_road_blocked": "road_blocked",
    }
    return m.get(concept_name, concept_name)


def load_concept_supervision(
    *, labels_dir: str | Path, split_csv: str | Path, split: str, concepts: list[ConceptSpec],
    buildings_csv: str | Path | None = None, roads_csv: str | Path | None = None,
) -> dict[str, list[ConceptRow]]:
    """Join classifier labels, reference boxes and paired-image paths for one split.

    Building boxes are needed only for building classifiers, road boxes only for road ones."""
    labels_dir = Path(labels_dir)
    split_idx = _load_split_index(split_csv)
    box_files = {"buildings": buildings_csv, "roads": roads_csv}
    boxes = {}
    for group in sorted({c.entity_group for c in concepts}):
        if box_files[group] is None:
            raise ValueError(f"Classifiers of {group} need {group}_csv")
        boxes[group] = load_reference_boxes(box_files[group])   # {(chip,eid): bbox}

    out: dict[str, list[ConceptRow]] = {}
    for c in concepts:
        label_csv = labels_dir / "per_entity" / split / f"{_label_stem(c.name)}.csv"
        if not label_csv.exists():
            raise FileNotFoundError(f"Label file not found: {label_csv}")
        bbox_lookup = boxes[c.entity_group]
        rows: list[ConceptRow] = []
        n_no_bbox = n_no_img = 0
        with label_csv.open(newline="") as f:
            for r in csv.DictReader(f):
                if r["label"] not in ("yes", "no"):
                    raise ValueError(f"{label_csv}: label must be yes or no, got {r['label']!r}")
                label = 1 if r["label"] == "yes" else 0
                chip_id, eid = r["chip_id"], r["entity_id"]
                bbox = bbox_lookup.get((chip_id, eid))
                if bbox is None:
                    n_no_bbox += 1
                    continue
                sp = split_idx.get(chip_id)
                if sp is None or not sp.get("pre_image_path"):
                    n_no_img += 1
                    continue
                rows.append(ConceptRow(
                    chip_id=chip_id, entity_id=eid,
                    bbox=tuple(float(v) for v in bbox), label=label,
                    pre_image_path=sp["pre_image_path"],
                    post_image_path=sp.get("post_image_path") if c.is_paired else None,
                ))
        out[c.name] = rows
        print(f"[concept-sup] {c.name}: {len(rows)} rows "
              f"(pos={sum(x.label for x in rows)}) "
              f"[dropped no_bbox={n_no_bbox} no_img={n_no_img}]", flush=True)
    return out
