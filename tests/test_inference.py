"""Image-inference checks with mocked models: no Satlas weights, GPU or images needed.
Checkpoint loading is tested in test_checkpoints.py."""
import copy
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')

from torch import nn
from siren.inference.buildings import CONFIG, BuildingPipeline
from siren.inference.pipeline import MODEL_KEYS, ScenePipeline, pipeline_from_assets
from siren.perception.loading import RoadConfig, RoadExtractor


def test_corrupt_asset_rejected_before_loading(tmp_path, monkeypatch):
    import siren.inference.pipeline as module
    path = tmp_path / 'base.pth'
    path.write_bytes(b'changed')
    monkeypatch.setattr(module, 'load_models', lambda *a: pytest.fail('must not load'))
    assets = {key: {'path': str(path), 'sha256': '0' * 64} for key in MODEL_KEYS}
    with pytest.raises(ValueError, match='hash mismatch'):
        pipeline_from_assets(assets, 'cpu')


# A mock detector: each bright rectangle in channel 0 of the view is one building, with its
# score in channel 1. Tiles are upsampled 2x, so rectangles map back to exact native boxes.
class Images:
    def __init__(self, tensors):
        self.tensors = tensors
        self.image_sizes = [tuple(tensors.shape[-2:])]


class Transform:
    image_mean, image_std = [0., 0., 0.], [1., 1., 1.]

    def __call__(self, images, targets):
        return Images(torch.stack(images)), None

    def postprocess(self, detections, image_sizes, original_sizes):
        return detections


class RoiHeads:
    score_thresh = 0.

    def __call__(self, fpn, proposals, image_sizes, targets):
        from scipy import ndimage
        view = proposals.tensors[0]
        labels, n = ndimage.label(view[0].numpy() > .5)
        masks = [torch.from_numpy(labels == k) for k in range(1, n + 1)]
        scores = torch.tensor([float(view[1][m].max()) for m in masks])
        stacked = torch.stack(masks)[:, None] if masks else torch.zeros(0, 1, 1024, 1024, dtype=torch.bool)
        return [dict(masks=stacked, scores=scores, labels=torch.ones(n, dtype=torch.long))], None


class Detector:
    def __init__(self):
        self.transform, self.roi_heads = Transform(), RoiHeads()

    def rpn(self, images, fpn, targets):
        return images, None


class MockModel:
    def __init__(self):
        self.det = Detector()
        self.fpn_calls = 0

    def eval(self):
        return self

    def fpn(self, image):
        self.fpn_calls += 1
        return {k: image for k in '0123'}


class Head(nn.Module):
    def forward(self, features):
        return torch.tensor([[0., 1.]]).expand(len(features), 2)


@pytest.fixture
def building_pipeline(monkeypatch):
    pytest.importorskip('scipy')  # the mock detector labels rectangles with scipy.ndimage
    import siren.inference.buildings as module
    monkeypatch.setattr(module, 'mask_to_polygons', lambda crop, **kw: [])
    model = MockModel()
    pipeline = BuildingPipeline(model, {q: Head() for q in ('is_damaged', 'is_destroyed')})
    pipeline.pooled = lambda fpn, boxes: torch.zeros(len(boxes), 6272)
    return pipeline, model


def image(rects):
    img = torch.zeros(3, 1024, 1024)
    for x0, y0, x1, y1, score in rects:
        img[0, y0:y1, x0:x1] = 1.
        img[1, y0:y1, x0:x1] = score
    return img


def boxes(record):
    return sorted(tuple(e['geometry']['bbox_xyxy']) for e in record['entities']['buildings'])


def test_tiling_maps_tiles_to_image_pixels_and_drops_cut_buildings(building_pipeline):
    pipeline, model = building_pipeline
    grid = [(10 + 40 * i, 10 + 40 * j, 30 + 40 * i, 30 + 40 * j, .9) for i in range(6) for j in range(6)]
    far = [(900, 900, 920, 920, .9), (900, 600, 920, 620, .9), (600, 900, 620, 920, .9),
           (700, 700, 720, 720, .9)]
    tile_only = (300, 300, 330, 330, .35)    # below the 0.4 full-view cutoff, above 0.3 for tiles
    cut = (500, 100, 530, 120, .9)           # crosses x = 512, an interior edge of tile (0, 0)
    border = (0, 480, 20, 500, .9)           # touches the image border inside tile (0, 0)
    small, tiny = (282, 282, 286, 286, .9), (292, 292, 295, 295, .9)   # 16 and 9 native pixels
    img = image(grid + far + [tile_only, cut, border, small, tiny])
    record = pipeline.run(img, img, chip_id='dense', split='test')[0]
    # 43 confident detections start tiling; only tile (0, 0) holds >= 25 of their centers (38).
    assert record['export']['tiled'] == 1
    assert model.fpn_calls == 3  # pre-image FPN for the full view and the tile, then post
    expected = sorted(tuple(map(float, r[:4])) for r in grid + far + [tile_only, cut, border, small])
    assert boxes(record) == expected  # one entity per building: duplicates merged, tiny dropped
    by_box = {tuple(e['geometry']['bbox_xyxy']): e for e in record['entities']['buildings']}
    assert by_box[tuple(map(float, tile_only[:4]))]['source']['view'] == 1
    assert by_box[tuple(map(float, tile_only[:4]))]['geometry']['area_px'] == 900.
    assert by_box[tuple(map(float, small[:4]))]['geometry']['area_px'] == 16.
    assert by_box[tuple(map(float, cut[:4]))]['source']['view'] == 0
    # In the tile itself, the cut building is dropped and the border building kept.
    tile = torch.nn.functional.interpolate(img[None, :, :512, :512], size=(1024, 1024),
                                           mode='bilinear', align_corners=False)[0].clamp(0, 1)
    tile_boxes = {tuple(d['box']) for d in pipeline.detect(tile, 0, 0, 512, 1, CONFIG.tile_score)[1]}
    assert (500., 100., 530., 120.) not in tile_boxes and (500., 100., 512., 120.) not in tile_boxes
    assert tuple(map(float, border[:4])) in tile_boxes
    assert tuple(map(float, tiny[:4])) not in tile_boxes


@pytest.mark.parametrize('n_confident,kept', [(40, True), (39, False)])
def test_tiling_trigger_sets_cutoffs_even_without_a_qualifying_tile(n_confident, kept,
                                                                    building_pipeline):
    pipeline, _ = building_pipeline
    # 8 x 5 spread grid: every 512-pixel tile holds at most 12 box centers.
    grid = [(30 + 125 * i, 30 + 200 * j, 50 + 125 * i, 50 + 200 * j, .9) for j in range(5) for i in range(8)]
    middling = (980, 980, 1000, 1000, .45)
    img = image(grid[:n_confident] + [middling])
    record = pipeline.run(img, img, chip_id='spread', split='test')[0]
    assert record['export']['tiled'] == 0
    assert ((980., 980., 1000., 1000.) in boxes(record)) == kept
    assert len(boxes(record)) == n_confident + kept


def test_building_inputs_are_checked(building_pipeline):
    pipeline, _ = building_pipeline
    good = torch.zeros(3, 1024, 1024)
    for pre, match in [(torch.zeros(3, 512, 512), '1024'), (good.double(), 'float32'),
                       (good + 2, r'\[0, 1\]'), (good * float('nan'), r'\[0, 1\]')]:
        with pytest.raises(ValueError, match=match):
            pipeline.run(pre, pre, chip_id='a', split='test')
    empty, _, post = pipeline.run(good, good, chip_id='a', split='test')
    assert empty['entities']['buildings'] == [] and empty['export']['tiled'] == 0
    assert post is None and empty['schema'] == 'siren_scene'
    assert empty['export']['pipeline'] == 'siren'


class SceneModel:
    def __init__(self):
        self.concepts = [SimpleNamespace(name=n, is_paired=(n == 'is_road_blocked'))
                         for n in ('is_residential', 'surrounded_by_vegetation', 'is_road_blocked')]
        self.heads = {'is_road_blocked': lambda x: torch.stack((-x.sum(-1), x.sum(-1)), -1)}
        self.post_calls = 0

    def concept_logits(self, name, boxes, fpn):
        assert fpn == 'pre-fpn'
        shift = {'is_residential': 1., 'surrounded_by_vegetation': -1.}[name]
        return torch.stack((torch.zeros(len(boxes)), boxes[:, 0] / 100 + shift), 1)

    def fpn_levels(self, fpn):
        return fpn

    def road_head(self, feats, size):
        return torch.zeros(1, 1, *size)

    def fpn(self, image):
        self.post_calls += 1
        return 'post-fpn'


@pytest.mark.parametrize('n_buildings,n_roads', [(0, 0), (0, 2), (2, 0), (2, 3)])
def test_scene_scores_and_post_features_only_when_needed(n_buildings, n_roads):
    model = SceneModel()
    buildings = [{'geometry': {'bbox_xyxy': [10. * i, 2., 10. * i + 8, 15.]}, 'attributes': {}}
                 for i in range(n_buildings)]
    roads = [{'geometry': {'bbox_xyxy': [0., 0., 20., 30.]}, 'attributes': {}} for _ in range(n_roads)]

    def extract_roads(mask):
        assert mask.shape == (1024, 1024) and mask.dtype == bool and not mask.any()
        return copy.deepcopy(roads)

    runtime = SimpleNamespace(model=model, extraction=RoadConfig(road_thresh=.8),
                              extract_roads=extract_roads,
                              entity_features=lambda fpn, boxes, *a: torch.full(
                                  (len(boxes), 2), 1. if fpn == 'post-fpn' else 0.))
    building = SimpleNamespace(model=model, run=lambda *a, **kw: (
        {'entities': {'buildings': copy.deepcopy(buildings)}, 'export': {}}, 'pre-fpn',
        'post-fpn' if n_buildings else None))
    image = torch.zeros(3, 1024, 1024)
    record = ScenePipeline(building, runtime).run(image, image, chip_id='a', split='test')
    for i, entity in enumerate(record['entities']['buildings']):
        x = 10. * i / 100
        assert entity['attributes']['is_residential_score'] == pytest.approx(torch.sigmoid(torch.tensor(x + 1.)).item())
        assert entity['attributes']['surrounded_by_vegetation_score'] == pytest.approx(torch.sigmoid(torch.tensor(x - 1.)).item())
    # Blocked-road features: zeros before, ones after and their absolute difference.
    assert [r['attributes']['is_road_blocked_score'] for r in record['entities']['roads']] == \
        pytest.approx([torch.sigmoid(torch.tensor(8.)).item()] * n_roads)
    assert model.post_calls == int(n_roads > 0 and n_buildings == 0)
    assert record['export']['post_features_for_roads_only'] == (n_roads > 0 and n_buildings == 0)
    assert record['export']['road_extraction'] == {
        'road_thresh': .8, 'road_min_area': 1024, 'road_close_radius': 2,
        'road_min_length_px': 180., 'road_spur_prune_px': 30.}


def test_runtime_checks_settings_and_shared_encoder():
    encoder = object()
    det = SimpleNamespace(backbone=encoder, transform=SimpleNamespace(image_mean=[0.] * 3,
                                                                     image_std=[1.] * 3))
    model = SimpleNamespace(det=det, backbone=encoder, eval=lambda: model)
    runtime = RoadExtractor(model, extraction=RoadConfig(road_thresh=.8))
    assert runtime.model is model
    with pytest.raises(ValueError, match='RoadConfig'):
        RoadExtractor(model, extraction={'road_thresh': .8})
    with pytest.raises(ValueError, match='road_thresh'):
        RoadConfig(road_thresh=1.5)
    other = SimpleNamespace(det=det, backbone=object(), eval=lambda: other)
    with pytest.raises(ValueError, match='same object'):
        RoadExtractor(other, extraction=RoadConfig(road_thresh=.8))
