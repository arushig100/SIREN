"""Check that load_checkpoint strictly loads base.pth and the classifier files, and that the
shipped classifier and damage checkpoints hold exactly the expected weights.

A small stand-in for SharedPerceptionModel keeps the same module names, so no Satlas weights,
torchvision detector or GPU are needed. Tests of the shipped files are skipped when the
release assets are not installed."""
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip('torch')

from torch import nn
from siren.perception.loading import load_checkpoint
from siren.settings import RoadConfig

ROOT = Path(__file__).resolve().parents[1]
EXTRACTION = RoadConfig(road_thresh=.8)


class TinyModel(nn.Module):
    """Same layout as SharedPerceptionModel: an encoder shared with the detector, a road
    segmentation head and one projection and head per classifier."""

    def __init__(self, *, concepts, pretrained_backbone=True):
        super().__init__()
        assert not pretrained_backbone
        self.concepts = list(concepts)
        self.det = nn.Module()
        self.det.backbone = nn.Linear(2, 2)
        self.det.transform = SimpleNamespace(image_mean=[0.] * 3, image_std=[1.] * 3)
        self.backbone = self.det.backbone
        self.road_head = nn.Linear(2, 1)
        self.concept_proj = nn.ModuleDict({c.name: nn.Sequential(nn.Linear(4, 3))
                                           for c in self.concepts})
        self.heads = nn.ModuleDict({c.name: nn.Sequential(nn.Linear(3, 2))
                                    for c in self.concepts})


@pytest.fixture
def tiny(monkeypatch):
    module = ModuleType('siren.perception.model')
    module.BASE_CLASSIFIERS = ('is_road_blocked',)
    module.SharedPerceptionModel = TinyModel
    monkeypatch.setitem(sys.modules, 'siren.perception.model', module)


def classifier_state(name, value):
    return {f'{group}.{name}.0.{p}': torch.full(shape, value)
            for group, shapes in (('concept_proj', {'weight': (3, 4), 'bias': (3,)}),
                                  ('heads', {'weight': (2, 3), 'bias': (2,)}))
            for p, shape in shapes.items()}


def base_state():
    encoder = {'weight': torch.full((2, 2), 1.), 'bias': torch.full((2,), 2.)}
    state = {f'{prefix}.{k}': v.clone() for prefix in ('backbone', 'det.backbone')
             for k, v in encoder.items()}
    state.update({'road_head.weight': torch.full((1, 2), 3.), 'road_head.bias': torch.zeros(1)})
    state.update(classifier_state('is_road_blocked', 4.))
    return state


def save(path, state, config=None):
    torch.save({'model_state_dict': state, 'config': config or {'image_size': 1024}}, path)
    return path


def test_base_alone_loads_exactly_the_blocked_road_classifier(tmp_path, tiny):
    state = base_state()
    runtime = load_checkpoint(save(tmp_path / 'base.pth', state), extraction=EXTRACTION)
    model = runtime.model
    assert [c.name for c in model.concepts] == ['is_road_blocked']
    loaded = model.state_dict()
    assert loaded.keys() == state.keys()
    assert all(torch.equal(loaded[k], v) for k, v in state.items())
    assert not model.training


def test_classifier_files_are_added_in_registry_order(tmp_path, tiny):
    files = {'surrounded_by_vegetation': save(tmp_path / 'veg.pth',
                                              classifier_state('surrounded_by_vegetation', 6.)),
             'is_residential': save(tmp_path / 'res.pth', classifier_state('is_residential', 5.))}
    model = load_checkpoint(save(tmp_path / 'base.pth', base_state()), extraction=EXTRACTION,
                            classifier_files=files).model
    assert [c.name for c in model.concepts] == [
        'is_residential', 'is_road_blocked', 'surrounded_by_vegetation']
    expected = {**base_state(), **classifier_state('is_residential', 5.),
                **classifier_state('surrounded_by_vegetation', 6.)}
    loaded = model.state_dict()
    assert loaded.keys() == expected.keys()
    assert all(torch.equal(loaded[k], v) for k, v in expected.items())


def test_missing_and_unexpected_weights_are_rejected(tmp_path, tiny):
    state = base_state()
    state.pop('road_head.bias')
    with pytest.raises(RuntimeError, match='road_head.bias'):
        load_checkpoint(save(tmp_path / 'a.pth', state), extraction=EXTRACTION)
    # base.pth must not hold the residential classifier; it comes from its own file.
    state = {**base_state(), **classifier_state('is_residential', 5.)}
    with pytest.raises(RuntimeError, match='is_residential'):
        load_checkpoint(save(tmp_path / 'b.pth', state), extraction=EXTRACTION)


def test_classifier_files_must_hold_only_their_classifier(tmp_path, tiny):
    base = save(tmp_path / 'base.pth', base_state())
    other = save(tmp_path / 'other.pth', {**classifier_state('is_residential', 5.),
                                          'road_head.bias': torch.ones(1)})
    with pytest.raises(ValueError, match='only the is_residential'):
        load_checkpoint(base, extraction=EXTRACTION, classifier_files={'is_residential': other})
    blocked = save(tmp_path / 'blocked.pth', classifier_state('is_road_blocked', 9.))
    with pytest.raises(ValueError, match='already holds'):
        load_checkpoint(base, extraction=EXTRACTION, classifier_files={'is_road_blocked': blocked})
    unknown = save(tmp_path / 'unknown.pth', classifier_state('is_main_road', 9.))
    with pytest.raises(ValueError, match='Unknown classifier'):
        load_checkpoint(base, extraction=EXTRACTION, classifier_files={'is_main_road': unknown})


def test_conflicting_backbone_aliases_and_settings_are_rejected(tmp_path, tiny):
    state = base_state()
    state['det.backbone.bias'] = torch.zeros(2)
    with pytest.raises(ValueError, match='Conflicting'):
        load_checkpoint(save(tmp_path / 'a.pth', state), extraction=EXTRACTION)
    path = save(tmp_path / 'b.pth', base_state(), {'image_size': 512})
    with pytest.raises(ValueError, match='image_size=512'):
        load_checkpoint(path, extraction=EXTRACTION)
    with pytest.raises(ValueError, match='RoadConfig'):
        load_checkpoint(path, extraction={'road_thresh': .8})
    torch.save({'weights': base_state()}, tmp_path / 'c.pth')
    with pytest.raises(ValueError, match='model_state_dict'):
        load_checkpoint(tmp_path / 'c.pth', extraction=EXTRACTION)


@pytest.mark.parametrize('name,concept', [('residential.pth', 'is_residential'),
                                          ('vegetation.pth', 'surrounded_by_vegetation')])
def test_classifier_checkpoints_hold_exactly_the_loaded_tensors(name, concept):
    path = ROOT / 'assets/checkpoints' / name
    if not path.exists():
        pytest.skip('release assets are not installed')
    state = torch.load(path, map_location='cpu', weights_only=True)['model_state_dict']
    # The projection and MLP head that SharedPerceptionModel builds for one classifier.
    expected = {
        f'concept_proj.{concept}.0.weight': (64, 6272), f'concept_proj.{concept}.0.bias': (64,),
        f'heads.{concept}.0.weight': (32, 64), f'heads.{concept}.0.bias': (32,),
        f'heads.{concept}.3.weight': (2, 32), f'heads.{concept}.3.bias': (2,)}
    assert {k: tuple(v.shape) for k, v in state.items()} == expected


@pytest.mark.parametrize('name', ['damaged.pth', 'destroyed.pth'])
def test_damage_heads_hold_only_their_weights(name):
    from siren.inference.buildings import BuildingConceptHead, load_head
    path = ROOT / 'assets/checkpoints' / name
    if not path.exists():
        pytest.skip('release assets are not installed')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    assert set(saved) == {'model_state_dict'}
    assert set(saved['model_state_dict']) == set(BuildingConceptHead().state_dict())
    assert not load_head(path, 'cpu').training
