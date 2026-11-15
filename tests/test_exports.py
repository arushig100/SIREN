"""Check export scene selection, output locking, hashes, resumption and run settings, plus the
asset config, data-root and road mask clean-up checks that exports rely on."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import siren.inference.export as export_module
from siren.inference.export import (atomic_json, export, locked_output, select_rows, sha,
                                    validate_outputs)

ROOT = Path(__file__).resolve().parents[1]


def manifest(tmp_path, duplicate=False):
    path = tmp_path / 'split.csv'
    rows = [dict(chip_id='a', split='train', pre_image_path='${DM3_ROOT}/pre.png', post_image_path='${DM3_ROOT}/post.png'),
            dict(chip_id='b', split='test', pre_image_path='/data/pre.png', post_image_path='/data/post.png')]
    if duplicate:
        rows.append(rows[0])
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_selection_and_path_resolution(tmp_path):
    path = manifest(tmp_path)
    rows = select_rows(path, 'train', '/data', set())
    assert [r['chip_id'] for r in rows] == ['a']
    assert rows[0]['pre_image_path'] == '/data/pre.png'
    with pytest.raises(ValueError, match='outside'):
        select_rows(path, 'test', '/data', set(), {'a'})
    with pytest.raises(ValueError, match='Empty'):
        select_rows(path, 'train', '/data', {'a'})
    with pytest.raises(ValueError, match='Duplicate'):
        select_rows(manifest(tmp_path, True), 'train', '/data', set())


def test_run_settings_and_lock_fail_closed(tmp_path):
    settings = {'scene_ids': ['a'], 'split': 'test'}
    atomic_json(tmp_path / '_run_settings.json', settings)
    with locked_output(tmp_path, settings):
        assert validate_outputs(tmp_path, settings) == {}
        with pytest.raises(RuntimeError, match='Another'):
            with locked_output(tmp_path, settings):
                pass
    with pytest.raises(ValueError, match='different settings'):
        with locked_output(tmp_path, dict(settings, split='val')):
            pass


def test_folder_with_files_from_another_export_rejected(tmp_path):
    (tmp_path / 'a.json').write_text('{}')
    with pytest.raises(ValueError, match='another export'):
        with locked_output(tmp_path, {}):
            pass


def test_resume_requires_intact_tracked_outputs(tmp_path):
    settings = {'scene_ids': ['a'], 'split': 'test'}
    path = tmp_path / 'a.json'
    atomic_json(path, {'chip_id': 'a', 'split': 'test'})
    with pytest.raises(ValueError, match='Untracked'):
        validate_outputs(tmp_path, settings)
    progress = {'a': sha(path)}
    atomic_json(tmp_path / '_progress.json', progress)
    assert validate_outputs(tmp_path, settings) == progress
    atomic_json(tmp_path / '_complete.json', progress)
    assert validate_outputs(tmp_path, settings) == progress
    path.write_text('{}')
    with pytest.raises(ValueError, match='hash mismatch'):
        validate_outputs(tmp_path, settings)


def test_incomplete_completion_rejected(tmp_path):
    atomic_json(tmp_path / '_complete.json', {})
    with pytest.raises(ValueError, match='completion'):
        validate_outputs(tmp_path, {'scene_ids': ['a'], 'split': 'test'})


def forbidden(*args):
    raise AssertionError('This run must not load the model')


def test_finished_resume_does_not_load_model(tmp_path, monkeypatch):
    settings = {'scene_ids': ['a'], 'split': 'test'}
    atomic_json(tmp_path / '_run_settings.json', settings)
    with locked_output(tmp_path, settings):
        atomic_json(tmp_path / 'a.json', {'chip_id': 'a', 'split': 'test'})
        progress = {'a': sha(tmp_path / 'a.json')}
        atomic_json(tmp_path / '_progress.json', progress)
        atomic_json(tmp_path / '_complete.json', progress)
    monkeypatch.setattr(export_module, 'prepare', lambda args: ({}, [{'chip_id': 'a'}], settings))
    monkeypatch.setattr(export_module, 'pipeline_from_assets', forbidden)
    export(SimpleNamespace(output=tmp_path, preflight=False))
    assert validate_outputs(tmp_path, settings) == progress


def test_failed_model_load_leaves_the_folder_reusable(tmp_path, monkeypatch):
    settings = {'scene_ids': ['a'], 'split': 'test'}

    def failing_load(*args):
        raise RuntimeError('no GPU')

    monkeypatch.setattr(export_module, 'prepare', lambda args: ({}, [{'chip_id': 'a'}], settings))
    monkeypatch.setattr(export_module, 'pipeline_from_assets', failing_load)
    with pytest.raises(RuntimeError, match='no GPU'):
        export(SimpleNamespace(output=tmp_path, preflight=False, device='cpu'))
    assert not (tmp_path / '_run_settings.json').exists()
    # A rerun with other settings may use the same folder.
    monkeypatch.setattr(export_module, 'prepare',
                        lambda args: ({}, [{'chip_id': 'a'}], dict(settings, split='val')))
    with pytest.raises(RuntimeError, match='no GPU'):
        export(SimpleNamespace(output=tmp_path, preflight=False, device='cpu'))


def test_preflight_does_not_create_outputs(tmp_path, capsys, monkeypatch):
    out = tmp_path / 'unused'
    monkeypatch.setattr(export_module, 'prepare', lambda args: ({}, [{'chip_id': 'a'}], {}))
    monkeypatch.setattr(export_module, 'pipeline_from_assets', forbidden)
    export(SimpleNamespace(output=out, preflight=True))
    assert not out.exists()
    assert json.loads(capsys.readouterr().out)['validated_scenes'] == 1


def test_run_settings_record_assets_sources_and_geometry_libraries(tmp_path):
    # prepare() reads the images and records the versions of these libraries.
    for module in ('PIL', 'torchvision', 'cv2', 'scipy', 'skimage'):
        pytest.importorskip(module)
    from PIL import Image
    from siren.inference.export import prepare
    from siren.inference.pipeline import ASSETS_SCHEMA, CLASSIFIER_ASSETS
    assets = tmp_path / 'assets'
    assets.mkdir()
    for name in ('pre.png', 'post.png', 'mask.png'):
        Image.new('RGB', (1024, 1024)).save(tmp_path / name)
    (assets / 'split.csv').write_text(
        'chip_id,split,pre_image_path,post_image_path,road_intact_mask_path,'
        'road_flooded_mask_path,road_debris_covered_mask_path\n'
        'a,test,${DM3_ROOT}/pre.png,${DM3_ROOT}/post.png,${DM3_ROOT}/mask.png,,\n')
    (assets / 'unreadable_scenes.json').write_text(json.dumps(
        {'split': 'val', 'reason': 'test', 'excluded_chip_ids': ['z']}))
    files = {'split_csv': 'split.csv', 'unreadable_scenes': 'unreadable_scenes.json'}
    for key in ('base', 'damaged', 'destroyed', *CLASSIFIER_ASSETS):
        (assets / key).write_bytes(key.encode())
        files[key] = key
    config = tmp_path / 'assets.json'
    config.write_text(json.dumps({'schema': ASSETS_SCHEMA, **{
        key: {'path': f'assets/{name}', 'sha256': sha(assets / name)} for key, name in files.items()}}))
    args = SimpleNamespace(assets=config, data_root=tmp_path, split='test', scene_ids=None,
                           device='cpu', preflight=True)
    _, rows, settings = prepare(args)
    assert [r['chip_id'] for r in rows] == settings['scene_ids'] == ['a']
    assert rows[0]['road_intact_mask_path'] == str(tmp_path / 'mask.png')
    assert rows[0]['road_flooded_mask_path'] == ''
    assert set(settings['assets']) == {'schema', *files}
    assert set(settings['runtime']) == {'torch', 'torchvision', 'numpy', 'scipy', 'scikit-image',
                                        'opencv', 'shapely', 'device'}
    sources = {Path(p).relative_to(ROOT / 'src').as_posix() for p in settings['source_sha256']}
    assert {'siren/inference/export.py', 'siren/perception/loading.py',
            'siren/segmentation/road/utils.py', 'siren/inference/pipeline.py',
            'siren/perception/train.py', 'siren/programs/engine.py'} <= sources
    # Preflight also opens the listed road masks.
    (tmp_path / 'mask.png').write_bytes(b'not an image')
    with pytest.raises(OSError):
        prepare(args)
    prepare(SimpleNamespace(**{**vars(args), 'preflight': False}))
    (assets / 'damaged').write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash mismatch: damaged'):
        prepare(args)


def test_asset_paths_follow_configuration_not_working_directory(tmp_path, monkeypatch):
    from siren.inference.pipeline import load_assets
    keys = ('base', 'damaged', 'destroyed', 'split_csv', 'unreadable_scenes',
            'residential', 'vegetation')
    original = {'schema': 'siren_assets',
                **{key: {'path': f'assets/{key}', 'sha256': 'a' * 64} for key in keys}}
    folder = tmp_path / 'configuration'
    folder.mkdir()
    config = folder / 'model.json'
    config.write_text(json.dumps(original))
    monkeypatch.chdir(tmp_path)
    assets = load_assets(config)
    for key in keys:
        assert assets[key] == {'path': str(folder / f'assets/{key}'), 'sha256': 'a' * 64}
    assert json.loads(config.read_text()) == original


@pytest.mark.parametrize('field,value,match', [
    ('split_csv', {'path': '', 'sha256': 'a'*64}, 'concrete asset path'),
    ('base', {'path': '${UNSET}/model.pth', 'sha256': 'a'*64}, 'concrete asset path'),
    ('base', {'path': 'model.pth', 'sha256': 'replace me'}, 'SHA256'),
    ('base', None, 'Missing asset record'),
    ('schema', 'unsupported_assets', 'Unsupported assets schema'),
])
def test_invalid_asset_configuration_is_rejected_before_loading(tmp_path, field, value, match):
    from siren.inference.pipeline import ASSET_KEYS, load_assets
    config = tmp_path / 'model.json'
    data = {'schema': 'siren_assets',
            **{key: {'path': key, 'sha256': 'a'*64} for key in ASSET_KEYS}}
    data[field] = value
    config.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=match):
        load_assets(config)


def test_missing_data_root_gives_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv('DM3_ROOT', raising=False)
    from siren.files import expand_data_path
    with pytest.raises(ValueError, match='Set DM3_ROOT'):
        expand_data_path('${DM3_ROOT}/DisasterM3_Bench/test_images/a.png')
    assert expand_data_path('') == ''
    assert expand_data_path('${DM3_ROOT}/a.png', '/data') == '/data/a.png'


@pytest.mark.parametrize('size,kept', [(1023, False), (1024, True), (1025, True)])
def test_road_clean_up_keeps_pieces_of_exactly_the_minimum_area(size, kept):
    import numpy as np
    from siren.segmentation.road.utils import clean_road_mask
    mask = np.zeros((40, 100), bool)
    mask.ravel()[:size] = True
    assert clean_road_mask(mask, min_area=1024, close_radius=0).any() == kept
