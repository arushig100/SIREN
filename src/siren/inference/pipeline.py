"""Scene inference: buildings with damage, destruction, residential and vegetation scores, and
road segments with blockage scores. load_pipeline reads configs/assets.json and loads the model."""
import json
from dataclasses import asdict
from pathlib import Path

import torch

from siren.files import sha256
from siren.settings import ROADS

ASSETS_SCHEMA = 'siren_assets'
# Asset records required in configs/assets.json.
ASSET_KEYS = ('base', 'damaged', 'destroyed', 'split_csv', 'unreadable_scenes',
              'residential', 'vegetation')
MODEL_KEYS = ('base', 'damaged', 'destroyed', 'residential', 'vegetation')
# Extra classifier files loaded with base.pth: asset key -> classifier name.
CLASSIFIER_ASSETS = {'residential': 'is_residential',
                     'vegetation': 'surrounded_by_vegetation'}


def load_assets(config_path):
    """Read the asset config; resolve paths relative to it and check the form of each hash."""
    config_path = Path(config_path).resolve()
    assets = json.loads(config_path.read_text())
    if not isinstance(assets, dict) or assets.get('schema') != ASSETS_SCHEMA:
        raise ValueError('Unsupported assets schema')

    def resolve(value, key):
        """Resolve one configured asset path relative to the config file."""
        if not isinstance(value, str) or not value.strip() or '${' in value:
            raise ValueError(f'Expected a concrete asset path: {key}')
        path = Path(value).expanduser()
        # Absolute paths are kept; relative paths are relative to the config file.
        return str(path if path.is_absolute() else (config_path.parent / path).resolve())

    for key in ASSET_KEYS:
        item = assets.get(key)
        if not isinstance(item, dict):
            raise ValueError(f'Missing asset record: {key}')
        digest = item.get('sha256')
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef'
                                                                   for c in digest):
            raise ValueError(f'Expected a lowercase SHA256: {key}')
        item['path'] = resolve(item.get('path'), key)
    return assets


def check_hashes(assets, keys=ASSET_KEYS):
    """Raise if a configured file differs from its recorded SHA-256."""
    for key in keys:
        if sha256(assets[key]['path']) != assets[key]['sha256']:
            raise ValueError(f'Asset hash mismatch: {key}')


def load_models(assets, device, classifier_files=None):
    """Load base.pth, any classifier files and both damage heads; return the building pipeline
    and the road extractor.

    classifier_files is passed to siren.perception.loading.load_checkpoint. The road extractor
    extracts roads and pools road features; the building pipeline applies the damage heads and
    their thresholds."""
    from siren.perception.loading import load_checkpoint
    from .buildings import BuildingPipeline, load_head
    # The segmentation cutoff defines road geometry; blockage decisions are made separately by
    # the query programs.
    roads = load_checkpoint(assets['base']['path'], extraction=ROADS,
                            classifier_files=classifier_files, device=device)
    heads = {concept: load_head(assets[key]['path'], device)
             for concept, key in [('is_damaged', 'damaged'), ('is_destroyed', 'destroyed')]}
    return BuildingPipeline(roads.model, heads), roads


def pipeline_from_assets(assets, device):
    """Check the model hashes, then load base.pth (with its blocked-road classifier), the
    residential and vegetation classifiers and both damage heads."""
    check_hashes(assets, MODEL_KEYS)
    classifier_files = {name: assets[key]['path'] for key, name in CLASSIFIER_ASSETS.items()}
    return ScenePipeline(*load_models(assets, device, classifier_files))


def load_pipeline(config_path, device):
    """Read an asset config such as configs/assets.json and load the full SIREN model."""
    return pipeline_from_assets(load_assets(config_path), device)


class ScenePipeline:
    """Score building and road entities in paired 1024-pixel images."""

    def __init__(self, building, roads):
        """Connect building and road inference to the same image encoder."""
        if building.model is not roads.model:
            raise ValueError('Buildings and roads must share the same model')
        if roads.extraction != ROADS:
            raise ValueError('Road extraction must use the shared settings')
        self.building, self.roads, self.model = building, roads, roads.model
        concepts = {c.name: c for c in self.model.concepts}
        if not {'is_residential', 'surrounded_by_vegetation', 'is_road_blocked'} <= set(concepts):
            raise ValueError('The model needs the residential, vegetation and blocked-road '
                             'classifiers')
        # Residential and vegetation use pre-image features; road blockage compares both images.
        if any(concepts[n].is_paired for n in ('is_residential', 'surrounded_by_vegetation')):
            raise ValueError('Residential and vegetation heads require pre-image features only')
        self.blocked = concepts['is_road_blocked']
        if not self.blocked.is_paired:
            raise ValueError('Road blockage requires paired features')

    @torch.inference_mode()
    def run(self, pre, post, *, chip_id, split, disaster_id=''):
        """Return entity IDs, geometry in image pixels and classifier probabilities.

        Buildings also store damage and destruction decisions at the counting thresholds.
        Query programs apply their own thresholds to the probabilities.
        """
        # Building extraction also returns the full-image features, which are reused below.
        record, fpn_pre, fpn_post = self.building.run(
            pre, post, chip_id=chip_id, split=split, disaster_id=disaster_id)
        buildings = record['entities']['buildings']
        if buildings:
            # Building boxes already lie inside the 1024-pixel image.
            boxes = torch.tensor([e['geometry']['bbox_xyxy'] for e in buildings],
                                 device=pre.device, dtype=torch.float32)
            residential = self.model.concept_logits('is_residential', boxes,
                                                    fpn_pre).softmax(1)[:, 1].cpu().tolist()
            vegetation = self.model.concept_logits('surrounded_by_vegetation', boxes,
                                                   fpn_pre).softmax(1)[:, 1].cpu().tolist()
            for entity, res, veg in zip(buildings, residential, vegetation):
                entity['attributes'].update(is_residential_score=res,
                                            surrounded_by_vegetation_score=veg)
        # Extract road geometry from the pre-disaster image; the decoder output is 1024x1024.
        prob = self.model.road_head(self.model.fpn_levels(fpn_pre), (1024, 1024)).sigmoid()
        roads = self.roads.extract_roads((prob[0, 0] >= self.roads.extraction.road_thresh).cpu().numpy())
        post_for_roads_only = bool(roads) and fpn_post is None
        if roads:
            # A scene without buildings has no post-image features yet.
            if fpn_post is None:
                fpn_post = self.model.fpn(post[None])
            boxes = torch.tensor([r['geometry']['bbox_xyxy'] for r in roads],
                                 device=pre.device, dtype=torch.float32)
            # Projected pre and post features of each road box, (N, 64) each; the head reads
            # [pre, post, |post - pre|].
            f = self.roads.entity_features(fpn_pre, boxes, self.blocked)
            g = self.roads.entity_features(fpn_post, boxes, self.blocked)
            scores = self.model.heads[self.blocked.name](torch.cat(
                (f, g, (g - f).abs()), dim=-1)).softmax(-1)[:, 1].cpu().tolist()
            for entity, score in zip(roads, scores):
                entity['attributes']['is_road_blocked_score'] = score
        record['entities']['roads'] = roads
        record['export']['road_extraction'] = asdict(self.roads.extraction)
        record['export']['post_features_for_roads_only'] = post_for_roads_only
        return record
