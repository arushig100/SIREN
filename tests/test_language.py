"""Check compiler requests and responses with a stand-in provider."""
import json
from types import SimpleNamespace
import pytest
from siren.language.compiler import compile_question, execute_compiled, validate_response
from siren.programs.recipes import building_fraction


def test_program_validation_and_unsupported():
    assert validate_response('{"program":null,"reason":"missing concept: flooding"}')['program'] is None
    for text in ['{"program":null,"program":null,"reason":"x"}', '{"program":NaN,"reason":""}',
                 '{"program":"import os","reason":""}']:
        with pytest.raises(ValueError):
            validate_response(text)


def test_request_holds_only_the_question_and_empty_scene_is_undefined():
    calls = []
    def create(**kw):
        calls.append(kw)
        return SimpleNamespace(content=[SimpleNamespace(type='text', text=json.dumps({'program':building_fraction(), 'reason':''}))], model='fake', id='test', usage=SimpleNamespace(model_dump=lambda: {}))
    result = compile_question('What fraction of buildings are damaged?', model='fake', client=SimpleNamespace(messages=SimpleNamespace(create=create)))
    assert calls[0]['messages'] == [{'role':'user', 'content':'What fraction of buildings are damaged?'}]
    assert execute_compiled(result, {'entities':{'buildings':[]}})['status'] == 'undefined'


def test_schema_covers_exact_operation_fields():
    from siren.language.contract import response_schema
    from siren.programs.engine import OPERATIONS
    schema = response_schema()
    variants = schema['properties']['program']['anyOf'][0]['properties']['steps']['items']['anyOf']
    assert len(variants) == len(OPERATIONS)
    for variant in variants:
        op = variant['properties']['op']['enum'][0]
        assert set(variant['required']) == OPERATIONS[op][0] | {'id', 'op'}
        assert variant['additionalProperties'] is False


def test_supported_explanation_is_metadata_not_program_failure():
    obj = validate_response(json.dumps({'program': building_fraction(),
                                        'reason': 'Divide damaged buildings by all detected buildings.'}))
    assert obj['program'] == building_fraction()


def test_cli_passes_request_settings(tmp_path, monkeypatch):
    import sys
    from siren.language import __main__ as cli
    def create(**kw):
        return SimpleNamespace(content=[SimpleNamespace(type='text', text=json.dumps({'program': building_fraction(), 'reason': ''}))], model='fake', id='test', usage=SimpleNamespace(model_dump=lambda: {}))
    result = compile_question('What fraction of buildings are damaged?', model='fake', client=SimpleNamespace(messages=SimpleNamespace(create=create)))
    assert result['run_settings_sha256'] and result['program']['format'] == 'siren.program'
    calls = []
    def fake_compile(question, **kw):
        calls.append(kw)
        return result
    monkeypatch.setattr(cli, 'compile_question', fake_compile)
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'test')
    monkeypatch.setattr(sys, 'argv', ['language', '--question', 'What fraction?', '--model', 'fake', '--output', str(tmp_path / 'program.json')])
    cli.main()
    assert calls == [{'model': 'fake', 'provider': 'anthropic', 'base_url': None, 'structured': True}]
    assert json.loads((tmp_path / 'program.json').read_text())['program'] == building_fraction()


def test_cli_requires_an_api_key(tmp_path, monkeypatch, capsys):
    import sys
    from siren.language import __main__ as cli
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    monkeypatch.setattr(sys, 'argv', ['language', '--question', 'What fraction?', '--model', 'fake', '--output', str(tmp_path / 'program.json')])
    with pytest.raises(SystemExit):
        cli.main()
    assert 'Set ANTHROPIC_API_KEY' in capsys.readouterr().err
    assert not (tmp_path / 'program.json').exists()


def test_structured_output_requested_and_invalid_response_retained():
    from siren.language.compiler import CompilationError
    calls = []
    def create(**kw):
        calls.append(kw)
        return SimpleNamespace(content=[SimpleNamespace(type='text', text='not json')], model='fake',
                               id='bad', stop_reason='end_turn',
                               usage=SimpleNamespace(model_dump=lambda: {'output_tokens': 2}))
    with pytest.raises(CompilationError) as error:
        compile_question('Count damaged buildings.', model='fake',
                         client=SimpleNamespace(messages=SimpleNamespace(create=create)))
    assert calls[0]['output_config']['format']['type'] == 'json_schema'
    assert error.value.record['raw_response'] == 'not json'
    assert error.value.record['usage']['output_tokens'] == 2


@pytest.mark.parametrize('structured', [True, False])
def test_openai_transport_and_local_execution(structured):
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        message = SimpleNamespace(content=json.dumps({'program': building_fraction(), 'reason': ''}),
                                  refusal=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason='stop')],
                               model='test-model', id='test', usage=None)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result = compile_question('What fraction of buildings are damaged?', model='test-model',
                              provider='openai', client=client, structured=structured)
    assert calls[0]['messages'][1] == {'role': 'user', 'content': result['question']}
    assert calls[0]['response_format']['type'] == ('json_schema' if structured else 'json_object')
    assert result['provider'] == 'openai'
    assert execute_compiled(result, {'entities': {'buildings': []}})['status'] == 'undefined'


@pytest.mark.parametrize('stop,refusal', [('length', None), ('stop', 'Cannot answer')])
def test_openai_truncation_and_refusal_are_not_executed(stop, refusal):
    from siren.language.compiler import CompilationError
    def create(**kwargs):
        message = SimpleNamespace(content=json.dumps({'program': building_fraction(), 'reason': ''}),
                                  refusal=refusal)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=stop)],
                               model='test-model', id='test', usage=None)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with pytest.raises(CompilationError):
        compile_question('Count buildings', model='test-model', provider='openai', client=client)


def test_custom_model_uses_the_shared_schema_and_validation():
    from siren.language.compiler import CompilationError
    calls = []
    def generate(**kwargs):
        calls.append(kwargs)
        return json.dumps({'program': building_fraction(), 'reason': ''})
    result = compile_question('Fraction damaged?', model='local-model', generate=generate)
    assert result['provider'] == 'custom'
    assert set(calls[0]) == {'question', 'model', 'prompt', 'schema'}
    assert calls[0]['schema']['properties']['program']
    with pytest.raises(CompilationError):
        compile_question('Count buildings', model='local-model', generate=lambda **_: 'not JSON')


def test_anthropic_json_mode_still_validates_programs():
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type='text', text=json.dumps(
            {'program': building_fraction(), 'reason': ''}))], model='fake', id='test', usage=None)
    result = compile_question('Fraction damaged?', model='fake', structured=False,
                              client=SimpleNamespace(messages=SimpleNamespace(create=create)))
    assert 'output_config' not in calls[0]
    assert result['program'] == building_fraction()
