"""Check the API vision-language model runner with stand-in clients: no network, keys or charges."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from siren.evaluation import vlm

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = json.loads((ROOT / 'configs/vlm_prompts.json').read_text())
ANSWER = {'type': 'output_text', 'text': '{"count": 3, "exists": true}', 'annotations': []}


def gpt_response(*content):
    """A Responses API result as the SDK returns it, with a reasoning item before the message."""
    responses = pytest.importorskip('openai.types.responses')
    return responses.Response.model_validate({
        'id': 'resp_1', 'object': 'response', 'created_at': 0, 'model': 'gpt-6.1-sol',
        'status': 'completed', 'parallel_tool_calls': False, 'tool_choice': 'auto', 'tools': [],
        'output': [{'type': 'reasoning', 'id': 'rs_1', 'summary': []},
                   {'type': 'message', 'id': 'msg_1', 'role': 'assistant',
                    'status': 'completed', 'content': list(content)}]})


@pytest.fixture
def fake_openai(monkeypatch):
    """Record client settings, uploads and request bodies; tests queue the replies."""
    openai = pytest.importorskip('openai')
    calls = SimpleNamespace(clients=[], uploads=[], bodies=[], replies=[])

    def upload(file, purpose):
        calls.uploads.append(Path(file.name).name)
        return SimpleNamespace(id=f'file-{len(calls.uploads)}')

    def respond(**body):
        calls.bodies.append(body)
        return calls.replies.pop(0)

    def build(**settings):
        calls.clients.append(settings)
        return SimpleNamespace(files=SimpleNamespace(create=upload),
                               responses=SimpleNamespace(create=respond))

    monkeypatch.setattr(openai, 'OpenAI', build)
    monkeypatch.setenv('OPENAI_API_KEY', 'test')
    return calls


def scenes(folder, *chip_ids):
    rows = []
    for chip_id in chip_ids:
        row = {'chip_id': chip_id}
        for when in ('pre', 'post'):
            image = folder / f'{chip_id}_{when}.png'
            image.write_bytes(f'{chip_id} {when}'.encode())
            row[f'{when}_image_path'] = str(image)
        rows.append(row)
    return rows


def run_args(output, question, **extra):
    settings = dict(output=output, question=question, provider='openai', variant='direct',
                    effort=None, model=None, uploads=None)
    return SimpleNamespace(**{**settings, **extra})


def test_gpt_answers_come_from_message_items(tmp_path, fake_openai):
    fake_openai.replies += [gpt_response(ANSWER),
                            gpt_response({'type': 'refusal', 'refusal': 'I cannot help.'})]
    rows = scenes(tmp_path, 'a', 'b')
    args = run_args(tmp_path / 'damaged_buildings', 'damaged_buildings')
    vlm.run_requests(args, rows, PROMPTS, [], {})
    a, b = (json.loads((args.output / f'{chip_id}.json').read_text()) for chip_id in 'ab')
    assert a['answer']['count'] == 3 and a['error'] is None
    assert b['answer'] is None and b['error'] == 'Model refused: I cannot help.'
    with pytest.raises(RuntimeError, match='1 unanswered'):
        vlm.write_predictions(args, rows)
    assert not (args.output / 'predictions.csv').exists()
    # Paper settings: GPT-6.1 Sol, low effort, original-detail images, strict direct-count JSON.
    assert fake_openai.clients == [{'max_retries': 3, 'timeout': 1800}]
    body = fake_openai.bodies[0]
    content = body['input'][0]['content']
    assert body['model'] == 'gpt-6.1-sol' and body['reasoning'] == {'effort': 'low'}
    assert body['max_output_tokens'] == 32000 and body['text']['format']['strict'] is True
    assert content[1] == {'type': 'input_image', 'file_id': 'file-1', 'detail': 'original'}
    assert content[-1]['text'] == (PROMPTS['questions']['damaged_buildings'] + '\n' +
                                   PROMPTS['variants']['direct']['instruction'])


def test_questions_can_share_uploaded_images(tmp_path, fake_openai):
    rows = scenes(tmp_path, 'a')
    registry = tmp_path / 'uploads.json'
    for question in ('damaged_buildings', 'destroyed_buildings'):
        fake_openai.replies.append(gpt_response(ANSWER))
        vlm.run_requests(run_args(tmp_path / question, question, uploads=registry),
                         rows, PROMPTS, [], {})
    assert fake_openai.uploads == ['a_pre.png', 'a_post.png']
    assert fake_openai.bodies[1]['input'][0]['content'][3]['file_id'] == 'file-2'
    assert list(json.loads(registry.read_text())) == ['openai']


def test_resume_parses_saved_responses_without_requests(tmp_path, monkeypatch):
    rows = [{'chip_id': 'a'}, {'chip_id': 'b'}]
    (tmp_path / 'run_settings.json').write_text('{}')
    for row, content in zip(rows, (ANSWER, {**ANSWER, 'text': 'not JSON'})):
        # Saved paid responses whose answers were not parsed when they arrived.
        (tmp_path / f"{row['chip_id']}.json").write_text(json.dumps({
            'chip_id': row['chip_id'], 'question': 'damaged_buildings',
            'response': gpt_response(content).model_dump(),
            'answer': None, 'error': 'Expected a JSON object'}))
    monkeypatch.setitem(sys.modules, 'openai', None)  # Building a client would fail.
    args = run_args(tmp_path, 'damaged_buildings')
    vlm.run_requests(args, rows, PROMPTS, [], {})
    a, b = (json.loads((tmp_path / f'{chip_id}.json').read_text()) for chip_id in 'ab')
    assert a['answer']['count'] == 3 and a['error'] is None
    assert b['answer'] is None and b['error'].startswith('Expecting value')
    with pytest.raises(RuntimeError, match='1 unanswered'):
        vlm.write_predictions(args, rows)


def test_claude_refusal_is_an_error_even_with_a_valid_answer():
    message = pytest.importorskip('anthropic.types.beta').BetaMessage.model_validate({
        'id': 'msg_1', 'type': 'message', 'role': 'assistant', 'model': 'claude-opus-5-5',
        'content': [{'type': 'text', 'text': '{"count": 0, "exists": false}'}],
        'stop_reason': 'refusal', 'stop_sequence': None,
        'usage': {'input_tokens': 1, 'output_tokens': 1}})
    record = {'response': message.model_dump()}
    vlm.read_answer(record, 'anthropic', 'direct')
    assert record['answer'] is None and record['error'].startswith('Model refused')
    record['response']['stop_reason'] = 'end_turn'
    vlm.read_answer(record, 'anthropic', 'direct')
    assert record['answer']['count'] == 0 and record['error'] is None


def test_changed_request_settings_cannot_reuse_saved_answers(tmp_path, monkeypatch):
    (tmp_path / 'image.png').write_bytes(b'image')
    manifest = tmp_path / 'pairs.csv'
    manifest.write_text('chip_id,pre_image_path,post_image_path\na,image.png,image.png\n')

    def identity(provider='openai', effort=None):
        args = SimpleNamespace(pairs=manifest, prompts=ROOT / 'configs/vlm_prompts.json',
                               provider=provider, model=None, question='damaged_buildings', variant='direct',
                               effort=effort, examples=None)
        return vlm.prepare_inputs(args)[3]

    gpt, claude = identity(), identity('anthropic')
    assert (gpt['model'], gpt['effort']) == ('gpt-6.1-sol', 'low')
    assert (claude['model'], claude['effort']) == ('claude-opus-5-5', 'high')
    assert claude['request_settings']['thinking'] == {'type': 'adaptive'}
    assert claude['request_settings']['output_config']['effort'] == 'high'
    assert gpt['request_settings']['max_output_tokens'] == claude['request_settings']['max_tokens'] == 32000
    assert gpt['request_settings']['image']['detail'] == 'original'
    monkeypatch.setattr(vlm.importlib.metadata, 'version', lambda name: '0.0.1')
    other_sdk = identity()
    assert other_sdk['request_settings']['sdk'] == 'openai 0.0.1'
    assert len({b['settings_sha256'] for b in (gpt, claude, other_sdk)}) == 3
    output = tmp_path / 'output'
    output.mkdir()
    (output / 'run_settings.json').write_text(json.dumps(gpt))
    with pytest.raises(ValueError, match='different settings'):
        vlm.run_requests(run_args(output, 'damaged_buildings'), [], PROMPTS, [], other_sdk)


def test_parse_answer_rejects_invalid_counts():
    assert vlm.parse_answer('{"count": 0, "exists": false}', 'direct')['count'] == 0
    for text in ('{"count": -1, "exists": false}', '{"count": true, "exists": true}', 'refusal'):
        with pytest.raises(ValueError):
            vlm.parse_answer(text, 'direct')
    with pytest.raises(ValueError, match='listed entities'):
        vlm.parse_answer('{"count": 1, "exists": true, "items": []}', 'listing')


def test_interrupted_write_leaves_no_partial_file(tmp_path, monkeypatch):
    target = tmp_path / 'a.json'
    vlm.write_json(target, {'answer': None})
    write_text = Path.write_text

    def interrupted(path, text):
        write_text(path, text[:5])
        raise KeyboardInterrupt

    monkeypatch.setattr(Path, 'write_text', interrupted)
    with pytest.raises(KeyboardInterrupt):
        vlm.write_json(target, {'answer': {'count': 3}})
    assert json.loads(target.read_text()) == {'answer': None}
    assert [p.name for p in tmp_path.iterdir()] == ['a.json']
