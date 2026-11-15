"""Send image pairs and count questions to an OpenAI or Anthropic vision-language model.
Save the responses and parse counts for the shared evaluation code."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path

from siren.evaluation.external import pairs
from siren.files import sha256

# Paper defaults per provider; --model and --effort override them.
MODELS = {
    'openai': {
        'model': 'gpt-6.1-sol',
        'effort': 'low'
    },
    'anthropic': {
        'model': 'claude-opus-5-5',
        'effort': 'high'
    }
}


def schema_for(prompts, variant):
    """The answer schema sent to the provider, without minimum and maximum keywords."""
    # The examples variant uses the direct variant's answer format; parse_answer checks the
    # value ranges.
    schema = prompts['variants']['direct' if variant == 'examples' else variant]['schema']

    def clean(value):
        """Remove schema bounds that the provider does not support."""
        if isinstance(value, dict):
            return {k: clean(v) for (k, v) in value.items() if k not in ('minimum', 'maximum')}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value

    return clean(schema)


def parse_answer(text, variant):
    """Check a JSON answer and return its count. The answer needs a non-negative integer count,
    a Boolean exists and, for the listing variant, one item per entity with integer x and y
    pixel coordinates."""
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError('Expected a JSON object')
    n = value.get('count')
    if type(n) is not int or n < 0 or type(value.get('exists')) is not bool:
        raise ValueError('Expected a non-negative integer count and Boolean exists')
    if variant == 'listing':
        items = value.get('items')
        if not isinstance(items, list) or any(
            (not isinstance(x, dict) or set(x) != {'x', 'y'} or any(
                (type(v) is not int or not 0 <= v <= 1023 for v in x.values())) for x in items)):
            raise ValueError('Invalid entity coordinates')
        if len(items) != n:
            raise ValueError('Count does not match listed entities')
    return {'count': n}


def answer_text(provider, response):
    """Return the answer text of a saved response; a refusal is an error."""
    if provider == 'anthropic':
        text = ''.join(c.get('text', '') for c in response.get('content', []) if c.get('type') == 'text')
        if response.get('stop_reason') == 'refusal':
            raise ValueError('Model refused: ' + text)
        return text
    # Only message items hold the answer; reasoning items have content None.
    parts = [c for item in response.get('output') or [] if item.get('type') == 'message'
             for c in item.get('content') or []]
    refusals = [c.get('refusal') or '' for c in parts if c.get('type') == 'refusal']
    if refusals:
        raise ValueError('Model refused: ' + ' '.join(refusals))
    return ''.join(c.get('text', '') for c in parts if c.get('type') == 'output_text')


def read_answer(record, provider, variant):
    """Parse the record's saved response; a failure is kept as an error, never as a count."""
    try:
        record['answer'] = parse_answer(answer_text(provider, record['response']), variant)
        record['error'] = None
    except (ValueError, TypeError) as exc:
        record['answer'] = None
        record['error'] = str(exc)


def write_json(path, value):
    """Write through a temporary file in the same directory, so a crash never leaves a partial file."""
    text = json.dumps(value, indent=2)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        temporary.write_text(text)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_uploads(path):
    """Uploaded image file IDs, by provider and then image SHA-256."""
    return json.loads(path.read_text()) if path.exists() else {}


def request_body(provider, prompts, variant, parts, effort=None, model=None):
    """One request: OpenAI Responses API or Anthropic Messages API, with uploaded image files."""
    setting = MODELS[provider]
    model = model or setting['model']
    content = []
    for (kind, value) in parts:
        if provider == 'openai':
            content.append({
                'type': 'input_text',
                'text': value
            } if kind == 'text' else {
                'type': 'input_image',
                'file_id': value,
                'detail': 'original'
            })
        else:
            content.append({
                'type': 'text',
                'text': value
            } if kind == 'text' else {
                'type': 'image',
                'source': {
                    'type': 'file',
                    'file_id': value
                }
            })
    schema = schema_for(prompts, variant)
    effort = effort or setting['effort']
    if provider == 'openai':
        body = dict(model=model,
                    instructions=prompts['system'],
                    input=[{
                        'role': 'user',
                        'content': content
                    }],
                    reasoning={'effort': effort},
                    max_output_tokens=32000,
                    store=False,
                    text={
                        'format': {
                            'type': 'json_schema',
                            'name': 'count_answer',
                            'schema': schema,
                            'strict': True
                        }
                    })
        return body
    body = dict(model=model,
                system=prompts['system'],
                max_tokens=32000,
                messages=[{
                    'role': 'user',
                    'content': content
                }],
                thinking={'type': 'adaptive'},
                output_config={
                    'effort': effort,
                    'format': {
                        'type': 'json_schema',
                        'schema': schema
                    }
                },
                betas=['files-api-2025-04-14'])
    return body


def request_settings(provider, prompts, variant, effort=None, model=None):
    """Return what every request sends apart from texts and image IDs, plus the SDK version."""
    settings = request_body(provider, prompts, variant, [('image', '<file id>')], effort, model)
    settings.pop('instructions' if provider == 'openai' else 'system')
    messages = settings.pop('input' if provider == 'openai' else 'messages')
    settings['image'] = messages[0]['content'][0]
    try:
        # The provider name is also the name of its Python SDK.
        settings['sdk'] = provider + ' ' + importlib.metadata.version(provider)
    except importlib.metadata.PackageNotFoundError:
        settings['sdk'] = None  # Dry runs do not need the SDK.
    return settings


def parse_args():
    """Read the API provider, prompt, image pairs and output paths."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pairs', type=Path, required=True)
    ap.add_argument('--prompts', type=Path, required=True)
    ap.add_argument('--provider',
                    choices=MODELS,
                    required=True,
                    help='API provider: anthropic (Claude) or openai (GPT)')
    ap.add_argument('--model', help='Model identifier; defaults to the paper model for this provider')
    ap.add_argument('--question',
                    required=True,
                    help='Question name from configs/vlm_prompts.json, such as damaged_buildings')
    ap.add_argument('--variant',
                    choices=('direct', 'listing', 'examples'),
                    default='direct',
                    help='Prompt: direct count (paper default), entity listing or three examples; '
                    'listing and examples are for the validation prompt comparison and need '
                    'split=val rows')
    ap.add_argument('--effort',
                    choices=('low', 'medium', 'high'),
                    help='Reasoning effort (default: high for anthropic, low for openai, as in '
                    'the paper)')
    ap.add_argument('--examples',
                    type=Path,
                    help='examples variant only: CSV of three validation pairs with split (val) '
                    'and reference columns')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--uploads',
                    type=Path,
                    help='JSON file of uploaded image IDs that several runs can share; '
                    'defaults to uploads.json in the output directory')
    ap.add_argument('--execute',
                    action='store_true',
                    help='Send images to the selected provider; incurs API charges')
    return ap.parse_args()


def prepare_inputs(args):
    """Check the scenes, prompts and examples, and return them with the run settings."""
    rows = pairs(args.pairs)
    prompts = json.loads(args.prompts.read_text())
    if args.question not in prompts['questions']:
        raise ValueError('Question is not in the paper prompt set')
    examples = pairs(args.examples) if args.examples else []
    if args.variant == 'examples':
        if len(examples) != 3 or any((r.get('split') != 'val' for r in examples)):
            raise ValueError('Three-example prompting requires three validation scenes')
        if {r['chip_id'] for r in examples} & {r['chip_id'] for r in rows}:
            raise ValueError('Examples overlap the evaluation scenes')
        for row in examples:
            if int(row['reference']) < 0:
                raise ValueError('Invalid example count')
    elif examples:
        raise ValueError('Examples are used only by the examples variant')
    if args.variant != 'direct' and any((r.get('split') != 'val' for r in rows)):
        raise ValueError('Prompt comparisons must use validation scenes')
    effort = args.effort or MODELS[args.provider]['effort']
    model = args.model or MODELS[args.provider]['model']
    settings = request_settings(args.provider, prompts, args.variant, effort, model)
    run_settings = {
        'pairs_sha256': sha256(args.pairs),
        'prompts_sha256': sha256(args.prompts),
        'provider': args.provider,
        'question': args.question,
        'variant': args.variant,
        'effort': effort,
        'model': model,
        'request_settings': settings,
        'settings_sha256': hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest(),
        'examples_sha256': sha256(args.examples) if args.examples else None,
        'images': {
            r['chip_id']: {
                k: sha256(r[k])
                for k in ('pre_image_path', 'post_image_path')
            }
            for r in rows + examples
        }
    }
    return (rows, prompts, examples, run_settings)


def read_saved_result(path, chip_id, question):
    """Read a saved answer and check its scene, question and count."""
    saved = json.loads(path.read_text())
    if not isinstance(saved, dict) or saved.get('chip_id') != chip_id or saved.get('question') != question:
        raise ValueError(f'Saved response belongs to a different scene or question: {path}')
    if 'answer' not in saved:
        raise ValueError(f'Saved response has no answer field: {path}')
    answer = saved['answer']
    if answer is not None and (not isinstance(answer, dict) or
                              type(answer.get('count')) is not int or answer['count'] < 0):
        raise ValueError(f'Saved response has an invalid count: {path}')
    return saved


def run_requests(args, rows, prompts, examples, run_settings):
    """Resume saved requests, parse saved failures again, and store each response before parsing it."""
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / 'run_settings.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != run_settings:
            raise ValueError('Output folder was written with different settings; use a new folder')
    elif any(args.output.iterdir()):
        raise ValueError('Nonempty output folder has no run_settings.json; use a new folder')
    pending = []
    for row in rows:
        target = args.output / (row['chip_id'] + '.json')
        if not target.exists():
            pending.append(row)
            continue
        saved = read_saved_result(target, row['chip_id'], args.question)
        if saved['answer'] is None and saved.get('response') is not None:
            # Parse the paid response again instead of sending the request again.
            read_answer(saved, args.provider, args.variant)
            write_json(target, saved)
    write_json(manifest, run_settings)
    if not pending:
        return
    key = 'OPENAI_API_KEY' if args.provider == 'openai' else 'ANTHROPIC_API_KEY'
    if not os.environ.get(key):
        raise ValueError(f'Set {key} to send requests to {args.provider}')
    if args.provider == 'openai':
        from openai import OpenAI
        client = OpenAI(max_retries=3, timeout=1800)
    else:
        from anthropic import Anthropic
        client = Anthropic(max_retries=3, timeout=1800)
    registry = args.uploads or args.output / 'uploads.json'
    registry.parent.mkdir(parents=True, exist_ok=True)
    uploads = {}

    def image_id(path):
        """Reuse an uploaded image or upload it and save its identifier."""
        digest = sha256(path)
        if digest not in uploads:
            # Runs for other questions may have added this image to a shared registry.
            uploads.update(read_uploads(registry).get(args.provider, {}))
        if digest not in uploads:
            if args.provider == 'openai':
                with open(path, 'rb') as f:
                    response = client.files.create(file=f, purpose='vision')
            else:
                from PIL import Image
                with Image.open(path) as image:
                    media_type = Image.MIME[image.format]
                with open(path, 'rb') as f:
                    response = client.beta.files.upload(file=(Path(path).name, f, media_type))
            uploads[digest] = response.id
            registered = read_uploads(registry)
            registered[args.provider] = {**registered.get(args.provider, {}), **uploads}
            write_json(registry, registered)
        return uploads[digest]

    def image_parts(row, prefix=''):
        """Return the labelled before and after images of one pair as message parts."""
        return [('text', prefix + 'Image 1 (BEFORE):'), ('image', image_id(row['pre_image_path'])),
                ('text', prefix + 'Image 2 (AFTER):'), ('image', image_id(row['post_image_path']))]

    for row in pending:
        target = args.output / (row['chip_id'] + '.json')
        parts = []
        if examples:
            parts.append(('text', prompts['variants']['examples']['example_preamble'].format(
                question=prompts['questions'][args.question])))
            for (i, example) in enumerate(examples, 1):
                parts += image_parts(example, f'Example {i}, ')
                parts.append(('text', f"Reference answer: {int(example['reference'])}"))
            parts.append(('text', prompts['variants']['examples']['target_label']))
        parts += image_parts(row)
        parts.append(('text', prompts['questions'][args.question] + '\n' +
                      prompts['variants'][args.variant]['instruction']))
        body = request_body(args.provider, prompts, args.variant, parts, args.effort,
                            model=args.model)
        # Time the request only; image uploads happen earlier and are shared across questions.
        start = time.perf_counter()
        if args.provider == 'openai':
            response = client.responses.create(**body)
        else:
            response = client.beta.messages.create(**body)
        elapsed = time.perf_counter() - start
        record = {
            'chip_id': row['chip_id'],
            'question': args.question,
            'seconds': elapsed,
            'request_sha256': hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(),
            'response': response.model_dump(),
            'answer': None,
            'error': None
        }
        write_json(target, record)
        read_answer(record, args.provider, args.variant)
        write_json(target, record)


def write_predictions(args, rows):
    """Write predictions.csv only if every scene has a parsed answer."""
    (failures, predictions) = ([], [])
    for row in rows:
        saved = read_saved_result(args.output / (row['chip_id'] + '.json'),
                                  row['chip_id'], args.question)
        if saved['answer'] is None:
            failures.append(row['chip_id'])
        else:
            predictions.append({'chip_id': row['chip_id'], 'prediction': saved['answer']['count']})
    if failures:
        raise RuntimeError(f'{len(failures)} unanswered scenes; no complete prediction CSV written')
    with (args.output / 'predictions.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['chip_id', 'prediction'])
        writer.writeheader()
        writer.writerows(predictions)


def main():
    """Check inputs, optionally request predictions and write count results."""
    args = parse_args()
    (rows, prompts, examples, run_settings) = prepare_inputs(args)
    if not args.execute:
        uploaded = read_uploads(args.uploads or args.output / 'uploads.json').get(args.provider, {})
        images = {digest for image in run_settings['images'].values() for digest in image.values()}
        print(
            json.dumps(
                {
                    'requests': len(rows),
                    'uploads': len(images - set(uploaded)),
                    'execution': 'dry run; no provider contacted',
                    'settings': {
                        k: v
                        for (k, v) in run_settings.items() if k != 'images'
                    }
                },
                indent=2))
        return
    run_requests(args, rows, prompts, examples, run_settings)
    write_predictions(args, rows)


if __name__ == '__main__':
    main()
