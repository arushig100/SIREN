"""Check that training outputs, query answers and evaluation inputs connect."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from siren.evaluation.counts import align_predictions, score
from siren.evaluation.external import count_scene, pairs
from siren.evaluation.learning import counts
from siren.evaluation.vlm import request_body
from siren.programs.recipes import COUNT_QUERIES, blocked_roads_near_homes

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def road_trainer(monkeypatch):
    """Use a tiny stand-in model to exercise the real training and export code."""
    pytest.importorskip('satlaspretrain_models')
    import torch
    from siren.perception import train
    from siren.perception.data import ConceptRow
    from siren.concepts.registry import get_concepts

    concept = next(c for c in get_concepts() if c.name == 'is_road_blocked')

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.concepts = [concept]
            self.concept_proj = torch.nn.ModuleDict({concept.name: torch.nn.Linear(4, 2)})
            self.heads = torch.nn.ModuleDict({concept.name: torch.nn.Linear(6, 2)})

        def freeze_backbone(self):
            pass

        def fpn(self, image):
            return image

        def pooled(self, features, boxes, name):
            return boxes + features

    def load_image(path, size, device):
        if path == 'unreadable':
            raise OSError('Unreadable image')
        return torch.tensor(1.0 if path == 'pre' else 2.0, device=device)

    monkeypatch.setattr(train, 'load_image_chw', load_image)
    # Interleaved scenes check that exported IDs follow the cached feature order.
    rows = [ConceptRow('a', '1', (0, 0, 1, 1), 0, 'pre', 'post'),
            ConceptRow('b', '2', (1, 1, 3, 3), 1, 'pre', 'post'),
            ConceptRow('a', '3', (2, 2, 4, 4), 1, 'pre', 'post')]
    return train, Model(), concept, rows


@pytest.mark.parametrize('field,value', [('pre_image_path', 'unreadable'),
                                        ('post_image_path', 'unreadable'),
                                        ('post_image_path', None)])
def test_road_training_rejects_missing_images(road_trainer, field, value):
    from dataclasses import replace
    train, model, concept, rows = road_trainer
    with pytest.raises(ValueError, match=r'image (path )?for a\b'):
        train._precompute_pooled(model, [replace(rows[0], **{field: value})], concept, 'cpu', 4)


def test_road_training_exports_selected_checkpoint_scores(road_trainer, tmp_path, monkeypatch):
    import torch
    from dataclasses import replace
    from siren.concepts.calibrate_roads import read_road_scores, select_access
    train, model, concept, rows = road_trainer
    invalid = replace(rows[0], chip_id='excluded', pre_image_path='unreadable')
    manifest = tmp_path / 'unreadable_scenes.json'
    manifest.write_text(json.dumps({'split': 'val', 'reason': 'unreadable image',
                                    'excluded_chip_ids': ['excluded']}))
    monkeypatch.setattr(train, 'load_concept_supervision',
        lambda **kw: {concept.name: rows + ([invalid] if kw['split'] == 'val' else [])})
    args = SimpleNamespace(labels_dir='', split_csv='', roads_csv='',
        unreadable_scenes=str(manifest), head_lr_s2=.01, weight_decay=0., epochs_s2=3,
        batch_size_s2=2)
    train.stage2(args, model, 'cpu', tmp_path)
    units = json.loads((tmp_path / 'validation_road_scores.json').read_text())
    assert [(u['chip_id'], u['entity_id']) for u in units] == [('a', '1'), ('a', '3'), ('b', '2')]
    model.load_state_dict(torch.load(tmp_path / 'base.pth', weights_only=True)['model_state_dict'])
    model.eval()
    pre, post, labels, _ = train._precompute_pooled(model, rows, concept, 'cpu', 4)
    with torch.no_grad():
        proj = model.concept_proj[concept.name]
        features = train._combine(proj(torch.tensor(pre).float()), proj(torch.tensor(post).float()))
        expected = model.heads[concept.name](features).softmax(-1)[:, 1].tolist()
    assert [u['score'] for u in units] == pytest.approx(expected)
    assert [u['label'] for u in units] == labels.tolist()
    assert select_access(*read_road_scores(tmp_path / 'validation_road_scores.json'))['n_segments'] == 3
    # The validation list of unreadable scenes must not hide a broken training input.
    monkeypatch.setattr(train, 'load_concept_supervision', lambda **kw: {concept.name: rows + [invalid]})
    with pytest.raises(ValueError, match='Cannot read pre image for excluded'):
        train.stage2(args, model, 'cpu', tmp_path)


def test_prediction_join_uses_scene_ids(tmp_path):
    predictions, references = tmp_path / 'predictions.csv', tmp_path / 'references.csv'
    predictions.write_text('chip_id,prediction\nb,2\na,0\n')
    references.write_text('chip_id,count\na,1\nb,2\n')
    rows = align_predictions(predictions, references, 'damaged_buildings', 'baseline')
    assert score(rows)['damaged_buildings']['systems']['baseline']['count_mae'] == .5
    predictions.write_text('chip_id,prediction\na,1\n')
    with pytest.raises(ValueError, match='same scene IDs'):
        align_predictions(predictions, references, 'damaged_buildings', 'baseline')
    predictions.write_text('chip_id,prediction\na,1\na,1\n')
    with pytest.raises(ValueError, match='duplicated'):
        align_predictions(predictions, references, 'damaged_buildings', 'baseline')


def test_model_comparison_requires_matching_references():
    rows = [dict(question='q', system='SIREN', chip_id='a', reference=1, prediction=1),
            dict(question='q', system='baseline', chip_id='a', reference=2, prediction=1)]
    with pytest.raises(ValueError, match='Reference disagreement'):
        score(rows)
    with pytest.raises(ValueError, match='Duplicate'):
        score(rows[:1] * 2)


def test_count_command_writes_error_reduction_for_all_eight_questions(tmp_path, monkeypatch):
    from siren.evaluation import counts as count_scoring
    lines = ['question,chip_id,system,reference,prediction']
    for question in COUNT_QUERIES:
        lines += [f'{question},a,SIREN,2,1', f'{question},b,SIREN,0,0']
    predictions = tmp_path / 'predictions.csv'
    predictions.write_text('\n'.join(lines) + '\n')
    output = tmp_path / 'metrics.json'
    monkeypatch.setattr(sys, 'argv', ['counts', '--predictions', str(predictions),
                                      '--output', str(output)])
    count_scoring.main()
    result = json.loads(output.read_text())
    assert result['questions']['blocked_roads'] == {
        'n': 2, 'always_zero_mae': 1.0, 'systems': {'SIREN': {'count_mae': .5}}}
    assert result['count_error_reduction_vs_zero'] == {'SIREN': 50.0}
    # With fewer than eight questions, only per-question metrics are written.
    predictions.write_text('\n'.join(lines[:3]) + '\n')
    output.unlink()
    count_scoring.main()
    assert list(json.loads(output.read_text())) == ['questions']


def test_summary_score_weights_queries_equally():
    from siren.evaluation.counts import count_error_reduction_vs_zero
    # Four queries have perfect counts; four have twice the always-zero error.
    # Different reference scales must not change the equal weighting of queries.
    results = {
        question: {'always_zero_mae': i + 1,
                   'systems': {'model': {'count_mae': 0 if i < 4 else 2 * (i + 1)}}}
        for i, question in enumerate(COUNT_QUERIES)
    }
    assert count_error_reduction_vs_zero(results)['model'] == pytest.approx(0)


def test_count_command_scores_events_separately(tmp_path, monkeypatch):
    from siren.evaluation import counts as count_scoring
    predictions = tmp_path / 'external.csv'
    predictions.write_text('event,question,chip_id,system,reference,prediction\n'
                           'ian,destroyed_buildings,a,SIREN,2,1\n'
                           'ian,destroyed_buildings,a,Other,2,2\n'
                           'eaton,destroyed_buildings,a,SIREN,4,4\n'
                           'eaton,destroyed_buildings,a,Other,4,0\n')
    output = tmp_path / 'metrics.json'
    monkeypatch.setattr(sys, 'argv', ['counts', '--predictions', str(predictions),
                                      '--output', str(output)])
    count_scoring.main()
    events = json.loads(output.read_text())['events']
    assert events['ian']['destroyed_buildings']['systems'] == {
        'SIREN': {'count_mae': 1.0}, 'Other': {'count_mae': 0.0}}
    assert events['eaton']['destroyed_buildings']['systems']['Other'] == {'count_mae': 4.0}
    with pytest.raises(SystemExit):
        count_scoring.main()  # the output file exists


def test_count_command_scores_a_partial_export(tmp_path, monkeypatch, example_scene):
    from siren.evaluation import counts as count_scoring
    entities = tmp_path / 'entities'
    entities.mkdir()
    (entities / 'synthetic.json').write_text(json.dumps(example_scene))
    references = tmp_path / 'references.csv'
    references.write_text('chip_id,count\nsynthetic,2\nmissing,0\n')
    ids = tmp_path / 'ids.txt'
    ids.write_text('synthetic\n')
    output = tmp_path / 'metrics.json'
    command = ['counts', '--entities-dir', str(entities), '--references', str(references),
               '--question', 'damaged_buildings', '--output', str(output)]
    monkeypatch.setattr(sys, 'argv', command)
    with pytest.raises(SystemExit):
        count_scoring.main()
    monkeypatch.setattr(sys, 'argv', [*command, '--scene-ids', str(ids)])
    count_scoring.main()
    result = json.loads(output.read_text())['questions']['damaged_buildings']
    assert result['n'] == 1 and result['systems'] == {'SIREN': {'count_mae': 0.0}}


def test_baseline_road_questions_need_a_road_mask_for_every_reference_scene(tmp_path):
    pytest.importorskip('torch')
    pytest.importorskip('PIL')  # baselines.data imports it
    from siren.baselines import train
    split_index = {}
    for split in ('train', 'val', 'test'):
        folder = tmp_path / split
        folder.mkdir()
        (folder / 'blocked_roads.csv').write_text(f'chip_id,count\n{split}_a,0\n{split}_b,1\n')
        for chip in (f'{split}_a', f'{split}_b'):
            split_index[chip] = dict(split=split, disaster_id='d', pre_image_path='pre.png',
                                     post_image_path='post.png', has_road_mask=True)
    args = SimpleNamespace(image_size=1024, question='blocked_roads', train_subset=None,
                           train_chips=None, batch_size=2, num_workers=0)
    encoder = SimpleNamespace(transform=lambda size: None)
    loaders = train.build_loaders(args, encoder, tmp_path, split_index, frozenset(), {}, 'cpu')
    assert len(loaders['test'].dataset) == 2
    split_index['test_b']['has_road_mask'] = False
    with pytest.raises(ValueError, match='no road mask'):
        train.build_loaders(args, encoder, tmp_path, split_index, frozenset(), {}, 'cpu')


def test_baseline_road_scene_counts(monkeypatch):
    """Road questions train on every scene with a road mask (less the unreadable validation
    scene) and score only the 1,748 test scenes of the road reference files."""
    split_csv = ROOT / 'assets/metadata/split.csv'
    if not split_csv.exists() or not (ROOT / 'data/counts').exists():
        pytest.skip('release assets and data are not installed')
    pytest.importorskip('torch')
    pytest.importorskip('PIL')
    from siren.baselines import train
    from siren.baselines.data import load_split_index
    from siren.files import read_unreadable_scenes
    monkeypatch.setenv('DM3_ROOT', '/data')
    split_index = load_split_index(split_csv)
    with_mask = {s: sum(r['has_road_mask'] for r in split_index.values() if r['split'] == s)
                 for s in ('train', 'val', 'test')}
    assert with_mask == {'train': 4360, 'val': 1073, 'test': 1805}
    excluded = read_unreadable_scenes(ROOT / 'assets/metadata/unreadable_scenes.json')
    encoder = SimpleNamespace(transform=lambda size: None)
    for question in ('blocked_roads', 'damaged_buildings_without_clear_road'):
        args = SimpleNamespace(image_size=1024, question=question, train_subset=None,
                               train_chips=None, batch_size=2, num_workers=0)
        loaders = train.build_loaders(args, encoder, ROOT / 'data/counts', split_index,
                                      excluded, {}, 'cpu')
        assert {s: len(loader.dataset) for s, loader in loaders.items()} == {
            'train': 4360, 'val': 1072, 'test': 1748}


def test_learning_count_program_matches_benchmark():
    # An intact building exactly 50 px away keeps a destroyed building from being isolated;
    # one 51 px away does not, and a damaged neighbor is not intact.
    def building(entity_id, x, damage, destruction):
        ring = [[x, 0], [x + 20, 0], [x + 20, 20], [x, 20]]
        return {'id': entity_id, 'geometry': {'bbox_xyxy': [x, 0, x + 20, 20], 'polygon_xy': [ring],
                                              'area_px': 400},
                'attributes': {'is_damaged_score': damage, 'is_destroyed_score': destruction}}

    buildings = [building('near', 0, .9, .9), building('intact_at_50', 70, .1, .1),
                 building('far', 300, .9, .9), building('damaged_at_10', 330, .8, .1),
                 building('intact_at_51', 371, .1, .1)]
    scene = {'chip_id': 'synthetic', 'entities': {'buildings': buildings, 'roads': []}}
    damage = [b['attributes']['is_damaged_score'] for b in buildings]
    destruction = [b['attributes']['is_destroyed_score'] for b in buildings]
    result = counts(buildings, damage, destruction, .55, .60)
    assert result == {'damaged_buildings': 3, 'destroyed_buildings_without_intact_neighbor': 1}
    for question in result:
        assert count_scene(scene, question) == result[question]


def test_external_distance_uses_image_scale():
    program = blocked_roads_near_homes()
    relation = next(s for s in program['steps'] if s['op'] == 'relate')
    assert relation['distance_px'] == pytest.approx(131.072)
    assert relation['distance_px'] != 50


@pytest.mark.parametrize('provider', ['openai', 'anthropic'])
@pytest.mark.parametrize('effort', ['low', 'high'])
def test_vlm_model_selection(provider, effort):
    prompts = json.loads((ROOT / 'configs/vlm_prompts.json').read_text())
    body = request_body(provider, prompts, 'direct', [], effort=effort, model='example-vision-model')
    assert body['model'] == 'example-vision-model'
    if provider == 'openai':
        assert body['reasoning'] == {'effort': effort}
    else:
        assert body['thinking'] == {'type': 'adaptive'}
        assert body['output_config']['effort'] == effort


def test_vlm_resume_requires_original_run_settings(tmp_path):
    from siren.evaluation.vlm import run_requests

    args = SimpleNamespace(output=tmp_path, question='damaged_buildings')
    (tmp_path / 'scene.json').write_text('{}')
    with pytest.raises(ValueError, match='no run_settings.json'):
        run_requests(args, [{'chip_id': 'scene'}], {}, [], {'model': 'original'})
    assert not (tmp_path / 'run_settings.json').exists()
    (tmp_path / 'run_settings.json').write_text('{"model": "original"}')
    with pytest.raises(ValueError, match='different settings'):
        run_requests(args, [{'chip_id': 'scene'}], {}, [], {'model': 'changed'})


@pytest.mark.parametrize('change', [
    {'chip_id': 'other'}, {'question': 'destroyed_buildings'},
    {'answer': {'count': -1}}, {'answer': {'count': True}},
])
def test_vlm_rejects_mismatched_or_invalid_saved_results(tmp_path, change):
    from siren.evaluation.vlm import run_requests, write_predictions

    args = SimpleNamespace(output=tmp_path, question='damaged_buildings')
    rows = [{'chip_id': 'scene'}]
    (tmp_path / 'run_settings.json').write_text('{}')
    saved = {'chip_id': 'scene', 'question': 'damaged_buildings', 'answer': {'count': 2}, **change}
    (tmp_path / 'scene.json').write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='Saved response'):
        run_requests(args, rows, {}, [], {})
    with pytest.raises(ValueError, match='Saved response'):
        write_predictions(args, rows)
    assert not (tmp_path / 'predictions.csv').exists()


@pytest.mark.parametrize('image_format,media_type', [('PNG', 'image/png'), ('JPEG', 'image/jpeg')])
def test_vlm_upload_type_and_resume(tmp_path, monkeypatch, image_format, media_type):
    pytest.importorskip('PIL')
    from PIL import Image
    from siren.evaluation.vlm import run_requests, write_predictions

    image = tmp_path / 'image.dat'
    Image.new('RGB', (4, 4)).save(image, format=image_format)
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'test-key')
    args = SimpleNamespace(output=tmp_path / 'results', question='damaged_buildings',
                           provider='anthropic', variant='direct', effort=None,
                           model='example-vision-model', uploads=None)
    rows = [dict(chip_id='scene', pre_image_path=str(image), post_image_path=str(image))]
    prompts = json.loads((ROOT / 'configs/vlm_prompts.json').read_text())
    calls = []

    def upload(*, file):
        calls.append(('upload', file[2]))
        assert file[1].read() == image.read_bytes()
        return SimpleNamespace(id='image-id')

    def request(**body):
        calls.append(('request', body['model']))
        return SimpleNamespace(model_dump=lambda: {'content': [
            {'type': 'text', 'text': '{"count": 2, "exists": true}'}]})

    client = SimpleNamespace(beta=SimpleNamespace(files=SimpleNamespace(upload=upload),
                                                  messages=SimpleNamespace(create=request)))
    monkeypatch.setitem(sys.modules, 'anthropic', SimpleNamespace(Anthropic=lambda **kw: client))
    run_requests(args, rows, prompts, [], {})
    assert calls == [('upload', media_type), ('request', 'example-vision-model')]
    # A complete resume needs neither another upload nor API credentials.
    monkeypatch.setitem(sys.modules, 'anthropic', None)
    monkeypatch.delenv('ANTHROPIC_API_KEY')
    run_requests(args, rows, prompts, [], {})
    write_predictions(args, rows)
    assert list(csv.DictReader((args.output / 'predictions.csv').open())) == [
        {'chip_id': 'scene', 'prediction': '2'}]


@pytest.mark.parametrize('train_labels,val_labels,message', [
    ([0, 0], [0, 1], 'both classes'),
    ([], [0, 1], 'both classes'),
    ([0, 1], [-1, -1], 'no labeled validation'),
])
def test_damage_training_rejects_unusable_labels(tmp_path, train_labels, val_labels, message):
    pytest.importorskip('torch')
    pytest.importorskip('PIL')  # train_damage imports it
    import numpy as np
    from siren.concepts.train_damage import train_heads

    def features(labels):
        return dict(labels=np.array(labels, dtype=np.int8).reshape(-1, 1),
                    is_reference=np.ones(len(labels), dtype=bool),
                    features=np.zeros((len(labels), 2, 6272), dtype=np.float16), scenes=[])

    data = dict(train=features(train_labels), val=features(val_labels))
    with pytest.raises(ValueError, match=message):
        train_heads(SimpleNamespace(seeds='0', output=tmp_path), data, {}, ['is_damaged'],
                    'cpu', None)


def test_saved_entities_connect_to_count_scoring(tmp_path, example_scene):
    scene = example_scene
    scene['chip_id'] = 'synthetic'
    entities = tmp_path / 'entities'
    entities.mkdir()
    (entities / 'synthetic.json').write_text(json.dumps(scene))
    references = tmp_path / 'references.csv'
    references.write_text('chip_id,count\nsynthetic,2\n')
    output = tmp_path / 'metrics.json'
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src'), 'PYTHONDONTWRITEBYTECODE': '1'}
    subprocess.run([sys.executable, '-m', 'siren.evaluation.counts',
                    '--entities-dir', str(entities), '--references', str(references),
                    '--question', 'damaged_buildings', '--output', str(output)],
                   env=env, cwd=tmp_path, check=True, capture_output=True)
    assert json.loads(output.read_text())['questions']['damaged_buildings']['systems']['SIREN']['count_mae'] == 0


def test_image_pair_paths_are_relative_to_manifest(tmp_path):
    image = tmp_path / 'image.png'
    image.write_bytes(b'path check only')
    manifest = tmp_path / 'pairs.csv'
    with manifest.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['chip_id', 'pre_image_path', 'post_image_path'])
        writer.writerow(['a', 'image.png', 'image.png'])
    assert pairs(manifest)[0]['pre_image_path'] == str(image)


@pytest.mark.parametrize('provider', ['openai', 'anthropic'])
def test_vlm_dry_run_reads_relative_image_paths(tmp_path, monkeypatch, capsys, provider):
    from siren.evaluation import vlm

    image = tmp_path / 'image.png'
    image.write_bytes(b'file hashing check')
    manifest = tmp_path / 'pairs.csv'
    manifest.write_text('chip_id,pre_image_path,post_image_path\na,image.png,image.png\n')
    output = tmp_path / 'output'
    monkeypatch.setattr(sys, 'argv', ['vlm', '--pairs', str(manifest),
        '--prompts', str(ROOT / 'configs/vlm_prompts.json'), '--provider', provider,
        '--question', 'damaged_buildings', '--output', str(output)])

    vlm.main()

    result = json.loads(capsys.readouterr().out)
    assert result['requests'] == 1
    assert result['execution'] == 'dry run; no provider contacted'
    assert not output.exists()

    # Custom model settings must appear in the saved-run identity as well.
    monkeypatch.setattr(sys, 'argv', sys.argv + ['--model', 'example-vision-model', '--effort', 'medium'])
    vlm.main()
    custom = json.loads(capsys.readouterr().out)
    assert custom['settings']['model'] == 'example-vision-model'
    assert custom['settings']['effort'] == 'medium'
    assert custom['settings'] != result['settings']


def test_baseline_inference_writes_predictions_and_run_info(tmp_path, monkeypatch):
    question = 'destroyed_buildings'
    torch = pytest.importorskip('torch')
    pytest.importorskip('PIL')
    from PIL import Image
    from siren.evaluation import external as inference
    from siren.files import sha256

    image = tmp_path / 'image.png'
    Image.new('RGB', (4, 4)).save(image)
    manifest = tmp_path / 'pairs.csv'
    manifest.write_text('chip_id,pre_image_path,post_image_path\na,image.png,image.png\n')
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'best.pth').write_bytes(b'stub checkpoint')
    output = tmp_path / 'output'

    # Exercise file loading and output writing without pretrained model assets.
    monkeypatch.setattr(inference, 'load_baseline', lambda *args: (
        lambda *images: torch.tensor(2.2),
        lambda image: torch.zeros(3, 4, 4),
        {'question': question, 'encoder': 'stub'},
    ))
    monkeypatch.setattr(sys, 'argv', ['inference', '--pairs', str(manifest),
        '--baseline-run', str(run), '--question', question, '--device', 'cpu',
        '--output', str(output)])

    inference.main()

    with (output / 'predictions.csv').open() as stream:
        predictions = list(csv.DictReader(stream))
    assert predictions[0]['chip_id'] == 'a'
    assert predictions[0]['prediction'] == '2'
    run_info = json.loads((output / 'run_info.json').read_text())
    assert run_info['images']['a']['post_image_path'] == sha256(image)
    assert run_info['question'] == question


def test_baseline_loader_restores_saved_weights(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    from siren.baselines import encoders
    from siren.baselines.model import ChipCountModel
    from siren.evaluation.external import load_baseline

    class Stub(encoders.Encoder):
        feat_dim = 4

        def __init__(self, ckpt=None, image_size=8):
            super().__init__()
            self.linear = torch.nn.Linear(3, 4)

        def transform(self, image_size):
            return ('transform', image_size)

        def forward(self, x):
            return self.linear(x.mean((2, 3)))

    monkeypatch.setitem(encoders.ENCODERS, 'stub', ('stub', 'Stub'))
    stub_module = SimpleNamespace(Stub=Stub)
    monkeypatch.setitem(sys.modules, 'siren.baselines.encoders.stub', stub_module)
    run = tmp_path / 'run'
    run.mkdir()
    saved = ChipCountModel(Stub()).state_dict()
    torch.save({'model': saved}, run / 'best.pth')
    (run / 'config.json').write_text(json.dumps({'encoder': 'stub', 'image_size': 8}))
    (model, transform, _) = load_baseline(run, 'cpu')
    assert transform == ('transform', 8) and not model.training
    assert all(torch.equal(v, saved[k]) for (k, v) in model.state_dict().items())


def test_shared_trainer_defaults_match_paper(monkeypatch):
    pytest.importorskip('satlaspretrain_models')
    from siren.perception import train
    argv = ['train', '--stage', '1', '--output', 'outputs', '--split-csv', 'split.csv',
            '--building-entities-dir', 'entities', '--labels-dir', 'labels',
            '--roads-csv', 'roads.csv', '--detector-checkpoint', 'detector.pth']
    monkeypatch.setattr(sys, 'argv', argv)
    args = train.build_args()
    # 40,000 minibatches of two with four accumulation steps: 10,000 AdamW updates at an
    # effective batch of eight, after 1,000 warm-up minibatches.
    assert (args.num_iter, args.batch_size, args.grad_accum) == (40_000, 2, 4)
    assert args.warmup_iter == 1000
    assert (args.weight_decay, args.w_road) == (1e-4, 0.2)
    # The pretrained encoder learns at 5e-6, the detection and road heads at 2e-4.
    encoder = SimpleNamespace(name='encoder', requires_grad=True)
    head = SimpleNamespace(name='head', requires_grad=True)
    model = SimpleNamespace(named_parameters=lambda: [('det.backbone.satlas_model.w', encoder),
                                                      ('road_head.w', head)])
    assert train._split_lr_groups(model, args.backbone_lr, args.head_lr) == [
        {'params': [encoder], 'lr': 5e-6}, {'params': [head], 'lr': 2e-4}]
    # Stage 2 needs its checkpoint and labels, not the Stage 1 inputs.
    stage2 = ['train', '--stage', '2', '--output', 'outputs', '--split-csv', 'split.csv',
              '--labels-dir', 'labels', '--roads-csv', 'roads.csv',
              '--stage1-checkpoint', 'stage1_last.pth']
    monkeypatch.setattr(sys, 'argv', stage2)
    assert train.build_args().stage1_checkpoint == 'stage1_last.pth'


def test_efficiency_separates_warmup_timing_and_profiling(monkeypatch):
    torch = pytest.importorskip('torch')
    from siren.evaluation import efficiency
    model = torch.nn.Linear(4, 2, bias=False)
    tensors = [torch.ones(1, 4), torch.ones(1, 4)]
    calls, syncs, loads = [], [], []

    def predict(tensor, row):
        calls.append(row['chip_id'])
        return model(tensor)

    def load(i):
        loads.append(i)
        return tensors[i]

    clock = iter([10, 10.5, 20, 20.25])
    monkeypatch.setattr(efficiency.time, 'perf_counter', lambda: next(clock))
    seconds, flops, used = efficiency.measure(
        predict, [model], load, [{'chip_id': 'a'}, {'chip_id': 'b'}],
        lambda: syncs.append(True), profile_scenes=1)
    assert seconds == [.5, .25]
    assert calls == ['a', 'b', 'a', 'b', 'a']  # Warmup, timing, profiling.
    assert loads == [0, 1, 0, 1, 0]  # Each pair is loaded when it is used.
    assert len(syncs) == 5
    assert flops == [16] and sum(used.values()) == 8
    assert not model._forward_pre_hooks


def test_efficiency_counts_functional_and_attention_parameters():
    torch = pytest.importorskip('torch')
    from siren.evaluation import efficiency

    class Functional(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.child = torch.nn.Linear(4, 4)
            self.attention = torch.nn.MultiheadAttention(4, 2, batch_first=True)

        def forward(self, x):
            # Neither child.forward nor attention.out_proj.forward runs.
            x = torch.nn.functional.linear(x, self.child.weight, self.child.bias)
            return self.attention(x, x, x, need_weights=False)[0]

    model = Functional().eval()
    fastpath = torch.backends.mha.get_fastpath_enabled()
    _, flops, used = efficiency.measure(lambda x, row: model(x), [model],
                                        lambda i: torch.ones(1, 3, 4), [{'chip_id': 'a'}],
                                        lambda: None)
    assert sum(used.values()) == sum(p.numel() for p in model.parameters())
    # 480 = F.linear plus both attention projections; the fast path would hide the attention.
    assert flops[0] >= 480
    assert torch.backends.mha.get_fastpath_enabled() == fastpath


def test_efficiency_checks_baseline_questions_before_loading(tmp_path, monkeypatch):
    from siren.evaluation import efficiency
    runs = []
    # damaged_buildings appears twice and destroyed_buildings is missing.
    questions = [COUNT_QUERIES[0], COUNT_QUERIES[0], *COUNT_QUERIES[2:]]
    for (i, question) in enumerate(questions):
        runs.append(tmp_path / f'run{i}')
        runs[-1].mkdir()
        (runs[-1] / 'config.json').write_text(json.dumps({'question': question}))
    monkeypatch.setattr(efficiency, 'load_baseline', lambda *args: pytest.fail('model loaded'))
    monkeypatch.setattr(sys, 'argv', ['efficiency', '--pairs', str(tmp_path / 'pairs.csv'),
        '--baseline-runs', *map(str, runs), '--output', str(tmp_path / 'out.json')])
    with pytest.raises(ValueError, match='each paper query once'):
        efficiency.main()
    (runs[1] / 'config.json').write_text(json.dumps({'question': 'destroyed_buildings'}))
    assert efficiency.baseline_questions(runs) == list(COUNT_QUERIES)


def test_baseline_training_defaults_follow_paper_recipes(monkeypatch):
    pytest.importorskip('torch')
    pytest.importorskip('PIL')
    import contextlib
    import io
    from siren.baselines import train

    def parse(encoder, *extra, question='damaged_buildings'):
        monkeypatch.setattr(sys, 'argv', ['train', '--encoder', encoder, '--question', question,
            '--split-csv', 'split.csv', '--counts-dir', 'counts', '--output', 'out', *extra])
        return train.parse_args()[1]

    # Learning rate, microbatch, accumulation and input size of each paper baseline.
    recipes = {'satlas': (1e-4, 2, 16, 1024), 'dinov3': (1e-5, 1, 32, 1024),
               'remoteclip': (1e-5, 2, 16, 1022), 'skyclip': (1e-5, 2, 16, 1022)}
    for (encoder, expected) in recipes.items():
        args = parse(encoder)
        assert (args.lr, args.batch_size, args.accum, args.image_size) == expected
        assert (args.batch_size * args.accum, args.epochs, args.weight_decay) == (32, 25, 1e-4)
    args = parse('satlas', '--lr', '3e-5', '--batch-size', '4', '--accum', '8')
    assert (args.lr, args.batch_size, args.accum) == (3e-5, 4, 8)
    # Unknown question names are rejected.
    with contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit):
        parse('satlas', question='not_a_question')


def test_road_threshold_selection_rules(tmp_path, monkeypatch):
    from siren.concepts import calibrate_roads
    from siren.settings import QUERY_THRESHOLDS

    # Road access: macro-F1 is highest only between .44 and .46, where the grid holds the
    # settings value; perfectly separated scores tie everywhere and keep the first threshold.
    access = calibrate_roads.select_access([.46, .46, .44, .44, .44], [1, 1, 0, 0, 0])
    assert access['threshold'] == QUERY_THRESHOLDS['is_road_blocked'] == 0.45000000000000007
    assert access['val_macro_f1'] == 1
    first = calibrate_roads.ACCESS_GRID[3]  # 0.11, the first grid value above 0.1
    assert calibrate_roads.select_access([.9, .1], [1, 0])['threshold'] == first
    # Counting: the total error is 1 at every grid value except .45 (the only one between .44
    # and .46), so .43 and .47 tie and the smaller one is kept. The scene without detections
    # still counts toward the mean.
    count = calibrate_roads.select_count([([.44], 1), ([.46], 0), ([], 0)])
    assert count['threshold'] == .43 and count['val_count_mae'] == pytest.approx(1 / 3)
    assert calibrate_roads.select_count([([.9], 1), ([.2], 0)])['threshold'] == .45

    # Eligible scenes: validation scenes with road labels, without the excluded input.
    split = tmp_path / 'split.csv'
    split.write_text('chip_id,split,road_intact_mask_path,road_flooded_mask_path,'
                     'road_debris_covered_mask_path\n'
                     'a,val,a.png,,\nb,val,,b.png,\nbad,val,bad.png,,\nunlabelled,val,,,\n'
                     'other,test,other.png,,\n')
    manifest = tmp_path / 'unreadable_scenes.json'
    manifest.write_text(json.dumps({'split': 'val', 'reason': 'unreadable image',
                                    'excluded_chip_ids': ['bad']}))
    references = tmp_path / 'blocked_roads.csv'
    references.write_text('chip_id,count\na,1\nb,0\nbad,5\nunlabelled,5\n')
    entities = tmp_path / 'entities'
    entities.mkdir()
    for (chip, scores) in {'a': [.95, .5], 'b': [.6], 'unlabelled': [.9] * 5}.items():
        roads = [{'attributes': {'is_road_blocked_score': s}} for s in scores]
        (entities / f'{chip}.json').write_text(
            json.dumps({'chip_id': chip, 'split': 'val', 'entities': {'roads': roads}}))
    output = tmp_path / 'thresholds.json'
    monkeypatch.setattr(sys, 'argv', ['calibrate_roads', '--entities-dir', str(entities),
        '--references', str(references), '--split-csv', str(split),
        '--unreadable-scenes', str(manifest), '--output', str(output)])
    calibrate_roads.main()
    result = json.loads(output.read_text())['road_counting']
    assert result['n_val_scenes'] == 2 and result['val_count_mae'] == 0
    assert result['threshold'] == .61  # Zero error from .61 to .95; .61 is closest to .45.


def test_recorded_road_scores_give_the_road_access_threshold():
    from siren.concepts import calibrate_roads
    from siren.settings import QUERY_THRESHOLDS
    path = ROOT / 'data/validation_road_scores.json'
    if not path.exists():
        pytest.skip('data/validation_road_scores.json is not installed; it ships with the data release')
    access = calibrate_roads.select_access(*calibrate_roads.read_road_scores(path))
    assert access['threshold'] == QUERY_THRESHOLDS['is_road_blocked'] == 0.45000000000000007
    assert (access['n_segments'], access['n_blocked']) == (7487, 823)
    assert access['val_macro_f1'] == pytest.approx(0.8922756834115068)


def test_damage_calibration_uses_complete_supplied_validation_set(tmp_path, monkeypatch):
    from siren.concepts import calibrate_damage
    candidates = tmp_path / 'candidates'
    candidates.mkdir()
    # One confident detection in a small scene; one reference building for each property.
    row = [0, .9, 0, 0, 10, 10, 100, 0, .8, .8]
    (candidates / 'scene.json').write_text(json.dumps({'chip_id': 'scene', 'cands': [row]}))
    references = tmp_path / 'counts' / 'val'
    references.mkdir(parents=True)
    for question in ('damaged_buildings', 'destroyed_buildings'):
        (references / f'{question}.csv').write_text('chip_id,count\nscene,1\n')
    output = tmp_path / 'thresholds.json'
    monkeypatch.setattr(sys, 'argv', ['calibrate_damage', '--candidates', str(candidates),
        '--counts-dir', str(references.parent), '--output', str(output)])
    calibrate_damage.main()
    result = json.loads(output.read_text())
    assert result['n_val_scenes'] == 1
    assert result['damaged']['val_count_mae'] == result['destroyed']['val_count_mae'] == 0
    # Missing predictions must fail instead of silently reducing the evaluation set.
    for question in ('damaged_buildings', 'destroyed_buildings'):
        (references / f'{question}.csv').write_text('chip_id,count\nscene,1\nmissing,2\n')
    with pytest.raises(ValueError, match='Missing candidate'):
        calibrate_damage.main()


def test_damage_calibration_excludes_only_documented_unreadable_inputs(tmp_path, monkeypatch):
    from siren.concepts import calibrate_damage
    candidates = tmp_path / 'candidates'
    candidates.mkdir()
    (candidates / 'valid.json').write_text(json.dumps({
        'chip_id': 'valid', 'cands': [[0, .9, 0, 0, 10, 10, 100, 0, .8, .8]]}))
    references = tmp_path / 'counts' / 'val'
    references.mkdir(parents=True)
    for q in ('damaged_buildings', 'destroyed_buildings'):
        (references / f'{q}.csv').write_text('chip_id,count\nvalid,1\nunreadable,2\n')
    unreadable = tmp_path / 'unreadable_scenes.json'
    unreadable.write_text(json.dumps({'split': 'val', 'reason': 'unreadable image',
                                      'excluded_chip_ids': ['unreadable']}))
    output = tmp_path / 'thresholds.json'
    monkeypatch.setattr(sys, 'argv', ['calibrate_damage', '--candidates', str(candidates),
        '--counts-dir', str(references.parent), '--unreadable-scenes', str(unreadable),
        '--output', str(output)])
    calibrate_damage.main()
    result = json.loads(output.read_text())
    assert result['n_val_scenes'] == 1 and result['unreadable_scenes'] == ['unreadable']
    # An unrelated missing prediction must still fail.
    for q in ('damaged_buildings', 'destroyed_buildings'):
        with (references / f'{q}.csv').open('a') as stream:
            stream.write('unexpected,1\n')
    with pytest.raises(ValueError, match='Missing candidate'):
        calibrate_damage.main()


@pytest.mark.parametrize('bad_path', ['pre', 'post'])
def test_image_error_never_substitutes_next_scene(monkeypatch, bad_path):
    pytest.importorskip('torch')
    pd = pytest.importorskip('pandas')
    pytest.importorskip('PIL')
    import numpy as np
    from siren.segmentation.common.satellite_dataset import BuildingInstanceDataset
    ds = object.__new__(BuildingInstanceDataset)
    ds.df = pd.DataFrame([dict(chip_id='bad', disaster_id='event', pre_image_path='pre',
                              post_image_path='post'),
                          dict(chip_id='next', disaster_id='event', pre_image_path='next_pre',
                              post_image_path='next_post')])
    ds.image_size = 4
    ds.building_entities_dir = Path('unused')
    monkeypatch.setattr(Path, 'read_text', lambda self: '{}')
    calls = []
    def read(path):
        calls.append(path)
        if path == bad_path:
            raise OSError('unreadable image')
        return np.zeros((4, 4, 3), dtype=np.uint8)
    ds._read_image = read
    with pytest.raises(RuntimeError, match='Cannot load scene bad'):
        ds[0]
    assert not any(path.startswith('next') for path in calls)


def test_road_masks_are_read_or_raise(tmp_path):
    pytest.importorskip('PIL')
    import numpy as np
    from PIL import Image
    from siren.segmentation.common.satellite_dataset import read_road_mask
    mask = np.zeros((8, 8), np.uint8)
    mask[2, 3] = 255
    Image.fromarray(mask).save(tmp_path / 'intact.png')
    row = {'road_intact_mask_path': str(tmp_path / 'intact.png'), 'road_flooded_mask_path': '',
           'road_debris_covered_mask_path': float('nan')}
    target = read_road_mask(row, image_size=8)
    assert target.dtype == np.int64 and target.sum() == 1 and target[2, 3] == 1
    # A listed mask that cannot be read is an error, not an empty mask.
    row['road_flooded_mask_path'] = str(tmp_path / 'missing.png')
    with pytest.raises(RuntimeError, match='Cannot read road mask'):
        read_road_mask(row, image_size=8)


def test_stage1_unreadable_scenes_are_skipped_in_validation_only(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    pd = pytest.importorskip('pandas')
    pytest.importorskip('PIL')
    from siren.perception import data
    manifest = tmp_path / 'unreadable_scenes.json'
    manifest.write_text(json.dumps({'split': 'val', 'reason': 'unreadable image',
                                    'excluded_chip_ids': ['excluded']}))
    split_csv = tmp_path / 'split.csv'
    split_csv.write_text('chip_id,split,road_intact_mask_path,road_flooded_mask_path,'
                         'road_debris_covered_mask_path\n'
                         'keep,val,keep.png,,\nexcluded,val,,,\n')
    monkeypatch.setattr(data, 'BuildingInstanceDataset', lambda **kw:
                        SimpleNamespace(df=pd.DataFrame({'chip_id': ['keep', 'excluded']})))
    for split in ('train', 'val', 'test'):
        ds = data.MultiTaskChipDataset(split_csv=split_csv, building_entities_dir='unused',
                                      split=split, unreadable_scenes=manifest)
        expected = ['keep'] if split == 'val' else ['keep', 'excluded']
        assert ds.buildings.df.chip_id.tolist() == expected
        assert list(ds.road_rows) == (['keep'] if split == 'val' else [])


def test_damage_training_outputs_connect_to_learning_evaluation(tmp_path):
    pytest.importorskip('torch')
    pytest.importorskip('PIL')
    import numpy as np
    from siren.concepts.train_damage import train_heads
    from siren.evaluation.learning import apply_heads

    # Two labelled entities exercise both classifier classes and checkpoint loading.
    features = np.zeros((2, 2, 6272), dtype=np.float32)
    candidates = [[0, .9, 0, 0, 20, 20, 400, 0], [0, .9, 50, 0, 70, 20, 400, 0]]
    rows = dict(features=features, labels=np.array([[0, 0], [1, 1]], dtype=np.int8),
                is_reference=np.ones(2, dtype=bool), scenes=[('scene', 'event', candidates)])
    train_heads(SimpleNamespace(seeds='0', output=tmp_path, epochs=1),
                dict(train=rows, val=rows), {}, ['is_damaged', 'is_destroyed'], 'cpu', None)
    run = tmp_path / 'seed_0'
    assert (run / 'damaged.pth').is_file() and (run / 'destroyed.pth').is_file()
    assert len(json.loads((run / 'candidates_val/scene.json').read_text())['cands']) == 2

    cache = tmp_path / 'cache'
    cache.mkdir()
    np.save(cache / 'test_feats.npy', features)
    (cache / 'score_check_passed').write_text('ok\n')
    (cache / 'index.json').write_text(json.dumps(dict(shape=list(features.shape), chips=[
        dict(chip_id='scene', start=0, n=2, export_sha256='reference')])))
    apply_heads(SimpleNamespace(cache=cache, heads=run, output=run / 'test_scores.npz', device='cpu'))
    with np.load(run / 'test_scores.npz') as scores:
        assert scores['is_damaged'].shape == scores['is_destroyed'].shape == (2,)
        assert np.isfinite(scores['is_damaged']).all() and np.isfinite(scores['is_destroyed']).all()


def test_road_loss_stops_on_non_finite_logits():
    torch = pytest.importorskip('torch')
    from siren.segmentation.road.losses import roads_loss
    logits = torch.zeros(1, 1, 8, 8)
    logits[0, 0, 0, 0] = float('nan')
    with pytest.raises(FloatingPointError):
        roads_loss(logits, torch.zeros(1, 8, 8))
