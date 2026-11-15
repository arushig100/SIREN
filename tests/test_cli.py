"""Check the saved-program command line, the eight count queries on the example scene, and
the shipped asset and prompt configs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_saved_program_cli_matches_engine(tmp_path, example_scene):
    from siren.programs.engine import execute
    from siren.programs.recipes import COUNT_QUERIES, count_program
    scene = example_scene
    scene_file = tmp_path / 'synthetic.json'
    scene_file.write_text(json.dumps(scene))
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src'), 'PYTHONDONTWRITEBYTECODE': '1'}
    for query in COUNT_QUERIES:
        program_file = tmp_path / f'{query}-program.json'
        program_file.write_text(json.dumps({'reason': '', 'program': count_program(query)}))
        answer_file = tmp_path / f'{query}-answer.json'
        subprocess.run([sys.executable, '-B', '-m', 'siren.language',
                        '--compiled', str(program_file), '--scene', str(scene_file),
                        '--output', str(answer_file)], cwd=tmp_path, env=env,
                       check=True, capture_output=True, text=True)
        answer = json.loads(answer_file.read_text())
        assert answer['answer']['value'] == execute(count_program(query), scene)['value']
        assert answer['execution_inputs'][str(scene_file)] == hashlib.sha256(scene_file.read_bytes()).hexdigest()


def test_shipped_asset_config_names_the_seven_release_assets():
    pytest.importorskip('torch')  # siren.inference.pipeline imports torch
    from siren.inference.pipeline import ASSET_KEYS, load_assets
    assets = load_assets(ROOT / 'configs/assets.json')
    keys = {'base', 'damaged', 'destroyed', 'split_csv', 'unreadable_scenes', 'residential',
            'vegetation'}
    assert set(ASSET_KEYS) == keys
    assert set(assets) == {'schema', *keys}
    for key in keys:
        assert Path(assets[key]['path']).parent.parent == (ROOT / 'assets').resolve()


def test_all_eight_count_queries(example_scene):
    from siren.programs.engine import execute
    from siren.programs.recipes import count_program
    scene = example_scene
    expected = {'damaged_buildings': 2, 'destroyed_buildings': 1, 'blocked_roads': 0,
                'damaged_buildings_without_clear_road': 1, 'destroyed_buildings_without_intact_neighbor': 1, 'large_destroyed_buildings': 1,
                'destroyed_residential_buildings': 1, 'destroyed_buildings_surrounded_by_vegetation': 1}
    for name, count in expected.items():
        assert execute(count_program(name), scene)['value'] == count


def test_question_names_agree_across_programs_and_prompts():
    from siren.programs.recipes import COUNT_QUERIES, count_program

    prompts = json.loads((ROOT / 'configs/vlm_prompts.json').read_text())
    assert tuple(prompts['questions']) == COUNT_QUERIES
    for name in COUNT_QUERIES:
        assert count_program(name)['output'] == 'answer'


def test_saved_program_cli_reads_context(tmp_path, monkeypatch, example_scene):
    from siren.language import __main__ as cli
    from siren.programs.recipes import paper_example
    scene_file = tmp_path / 'scene.json'
    scene_file.write_text(json.dumps(example_scene))
    context_file = tmp_path / 'context.json'
    context_file.write_text(json.dumps({'valid_area_px': 10000, 'pixel_size_m': 2}))
    program_file = tmp_path / 'program.json'
    program_file.write_text(json.dumps({'reason': '', 'program': paper_example('density')}))
    command = ['language', '--compiled', str(program_file), '--output']
    monkeypatch.setattr(sys, 'argv', [*command, str(tmp_path / 'answer.json'),
                                      '--scene', str(scene_file), '--context', str(context_file)])
    cli.main()
    answer = json.loads((tmp_path / 'answer.json').read_text())
    # One destroyed building in 10,000 pixels of 2 m: 0.04 km².
    assert answer['answer']['value'] == 25
    assert set(answer['execution_inputs']) == {str(scene_file.resolve()), str(context_file.resolve())}
    monkeypatch.setattr(sys, 'argv', [*command, str(tmp_path / 'other.json'),
                                      '--context', str(context_file)])
    with pytest.raises(SystemExit):
        cli.main()
    assert not (tmp_path / 'other.json').exists()
