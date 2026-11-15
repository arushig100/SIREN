"""Shared settings for entity extraction, classifier decisions and query geometry.

The defaults are the settings used for the paper.
"""
from dataclasses import dataclass
import math
import numpy as np

IMAGE_SIZE = 1024
# Context added to each side of a box before pooling: (minimum pixels, fraction of the box's
# longest side); the larger of the two applies.
DEFAULT_CONTEXT_PAD = (30.0, 0.0)
CONTEXT_PADS = {
    "surrounded_by_vegetation": (80.0, 0.75),
}

# Decision thresholds selected on validation data. is_road_blocked uses the road-access
# threshold; blocked-road counting uses BLOCKED_COUNT_THRESHOLD.
QUERY_THRESHOLDS = {
    "is_damaged": 0.55,
    "is_destroyed": 0.60,
    "is_road_blocked": 0.45000000000000007,  # the 0.45 entry of ROAD_ACCESS_GRID, float rounding kept
    "is_residential": 0.45,
    "surrounded_by_vegetation": 0.52,
}
BLOCKED_COUNT_THRESHOLD = 0.83
# Query geometry in DisasterM3 pixels (0.8 m per pixel): near means within 50 pixels (40 m);
# large means a footprint of at least 2,000 square pixels (1,280 m²).
NEAR_DISTANCE_PX = 50
LARGE_AREA_PX = 2000
# Road-mask columns of the split file; a scene has road labels if any of them is set.
ROAD_MASK_COLUMNS = ("road_intact_mask_path", "road_flooded_mask_path",
                     "road_debris_covered_mask_path")

# Threshold grids searched in validation calibration, and the tie-breaking reference.
DAMAGE_THRESHOLD_GRID = np.round(np.arange(0.3, 0.91, 0.05), 2)
ROAD_ACCESS_GRID = np.arange(0.05, 1.0, 0.02)
ROAD_COUNT_GRID = [n / 100 for n in range(5, 100, 2)]
ROAD_COUNT_TIE_THRESHOLD = 0.45
CLASSIFIER_THRESHOLD_GRID = [round(t, 2) for t in np.arange(0.05, 0.96, 0.01)]


@dataclass(frozen=True)
class BuildingConfig:
    """Paper settings for building detection and damage scoring; sizes are in pixels."""
    image_size: int = IMAGE_SIZE
    tile_size: int = 512
    tile_stride: int = 256
    tile_view_start: int = 1  # tiles are views 1-9 in row order; view 0 is the full image
    mask_threshold: float = 0.5  # mask probability for building pixels
    min_mask_area: int = 16  # smallest kept mask, in native image pixels
    tile_edge_margin: int = 2  # masks this close (view pixels) to an inner tile edge are cut off
    # Validation-selected damage and destruction thresholds (calibrate_damage).
    damaged_threshold: float = QUERY_THRESHOLDS["is_damaged"]
    destroyed_threshold: float = QUERY_THRESHOLDS["is_destroyed"]
    detector_score_floor: float = 0.30  # lowest detection score the detector returns
    full_score: float = 0.40   # full-view cutoff once tiling is triggered
    tile_score: float = 0.30   # cutoff for detections from a 512-pixel tile
    base_score: float = 0.50   # cutoff for other scenes and for the tiling trigger
    tile_trigger: int = 40     # full-view detections scoring >= base_score that trigger tiling
    tile_min_buildings: int = 25  # confident box centers a tile needs to be processed
    context_pad: float = DEFAULT_CONTEXT_PAD[0]  # context added to each side of a box before pooling
    # Duplicate rule: IoU >= 0.5, or the overlap covers >= 60% of the smaller box.
    merge_iou: float = 0.50
    merge_containment: float = 0.60

    def tile_origins(self) -> list[tuple[int, int]]:
        """Top-left corners (x, y) of the overlapping tiles, in row order (nine with the defaults)."""
        offsets = range(0, self.image_size - self.tile_size + 1, self.tile_stride)
        return [(x, y) for y in offsets for x in offsets]


@dataclass(frozen=True)
class RoadConfig:
    """Road-mask extraction settings; lengths and areas are measured in pixels."""
    # The segmentation cutoff defines road geometry; it is separate from blockage scores.
    road_thresh: float = 0.8
    road_min_area: int = 1024  # drop road fragments smaller than this (pixels)
    road_close_radius: int = 2  # closing radius that fills small gaps
    road_min_length_px: float = 180.0
    road_spur_prune_px: float = 30.0  # prune skeleton spurs shorter than this

    def __post_init__(self):
        """Reject invalid extraction probabilities, lengths and areas."""
        _probability(self.road_thresh, "road_thresh")
        for name in ("road_min_area", "road_close_radius", "road_min_length_px",
                     "road_spur_prune_px"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


def _probability(value, name):
    """Return value as a float; raise unless it is finite and in [0, 1]."""
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and in [0, 1]") from exc
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return value


BUILDINGS = BuildingConfig()
ROADS = RoadConfig()
