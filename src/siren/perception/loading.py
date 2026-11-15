"""Load the shared model from saved files, turn road masks into road segments and pool
entity features."""
from __future__ import annotations

from collections.abc import Mapping

import torch

from siren.concepts.registry import get_concepts
from siren.segmentation.road.utils import make_road_entities
from siren.settings import IMAGE_SIZE, RoadConfig

# A saved config may record the input size; only 1024 is supported.
ARCHITECTURE = {"image_size": IMAGE_SIZE}


def _read_model_file(path):
    """Return the model_state_dict and the config (empty if absent) of a saved model file."""
    raw = torch.load(path, map_location="cpu", weights_only=True)
    state = raw.get("model_state_dict") if isinstance(raw, Mapping) else None
    if not isinstance(state, Mapping):
        raise ValueError(f"{path} has no model_state_dict")
    config = raw.get("config") or {}
    if not isinstance(config, Mapping):
        raise ValueError(f"{path}: the saved config must be a mapping")
    return dict(state), config


def load_checkpoint(path, *, extraction, device="cpu", classifier_files=None):
    """Build the shared model and strictly load base.pth and any classifier files.

    base.pth holds the encoder, building detector, road segmentation head and blocked-road
    classifier. classifier_files maps other classifier names to their own files, for example
    {"is_residential": "residential.pth"}; each file holds only that classifier's
    projection and head. The Satlas weights are not downloaded."""
    if not isinstance(extraction, RoadConfig):
        raise ValueError("Caller must supply a RoadConfig")
    state, config = _read_model_file(path)
    for name, value in ARCHITECTURE.items():
        if config.get(name) not in (None, value):
            raise ValueError(f"Unsupported checkpoint setting {name}={config[name]!r}")
    # The shared encoder is saved under two prefixes (backbone. and det.backbone.). Reject
    # copies that differ, because load_state_dict would silently keep only one of them.
    for key, value in state.items():
        if key.startswith("backbone."):
            alias = "det." + key
            if alias in state and not torch.equal(value, state[alias]):
                raise ValueError(f"Conflicting shared backbone aliases: {key}")
    files = dict(classifier_files or {})
    for name, file in files.items():
        prefixes = (f"concept_proj.{name}.", f"heads.{name}.")
        if any(key.startswith(prefixes) for key in state):
            raise ValueError(f"{path} already holds the {name} classifier")
        weights, _ = _read_model_file(file)
        if not weights or not all(key.startswith(prefixes) for key in weights):
            raise ValueError(f"{file} must hold only the {name} classifier")
        state.update(weights)
    # Imported here so that tests of this module run without torchvision and Satlas.
    from siren.perception.model import BASE_CLASSIFIERS, SharedPerceptionModel
    names = {*BASE_CLASSIFIERS, *files}
    # Build the classifiers in registry order.
    concepts = [c for c in get_concepts() if c.name in names]
    if len(concepts) != len(names):
        raise ValueError(f"Unknown classifier name in {sorted(files)}")
    model = SharedPerceptionModel(concepts=concepts, pretrained_backbone=False)
    model.load_state_dict(state, strict=True)
    # A buffer outside the state dict would keep its construction-time value.
    persistent = model.state_dict().keys()
    if any(name not in persistent for name, _ in model.named_buffers()):
        raise ValueError("Model has buffers that the checkpoint does not set")
    model.to(device).eval()
    return RoadExtractor(model, extraction=extraction)


class RoadExtractor:
    """Road extraction and entity features around a loaded model."""

    def __init__(self, model, *, extraction):
        """Attach a loaded model and check its extraction settings and shared encoder."""
        if not isinstance(extraction, RoadConfig):
            raise ValueError("Caller must supply a RoadConfig")
        self.model = model.eval()
        self.extraction = extraction
        if model.backbone is not model.det.backbone:
            raise ValueError("Detector and shared backbone must be the same object")
        transform = model.det.transform
        if any(transform.image_mean) or any(v != 1 for v in transform.image_std):
            raise ValueError("Satlas requires identity detector normalization")

    def entity_features(self, fpn, boxes, concept):
        """Pool 7x7 features around boxes with the classifier's context padding and project
        them: (N, 64)."""
        return self.model.project(concept.name, self.model.pooled(fpn, boxes, concept.name))

    def extract_roads(self, mask):
        """Turn a binary 1024 x 1024 road mask into road segments (centerline and box)."""
        cfg = self.extraction
        return make_road_entities(
            mask, min_area=cfg.road_min_area, close_radius=cfg.road_close_radius,
            min_length_px=cfg.road_min_length_px, spur_prune_px=cfg.road_spur_prune_px)
