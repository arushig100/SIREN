"""Compile a question, or reuse a saved program, and optionally run it on a scene record."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from .compiler import compile_question, execute_compiled, validate_response


def main():
    """Compile a question or reuse a saved program, and optionally run it on a scene record."""
    ap = argparse.ArgumentParser(description=__doc__)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument('--question', help='Question to translate into a program')
    source.add_argument('--compiled',
                        type=Path,
                        help='Reuse a saved compiler result without an API call')
    ap.add_argument('--model', help='Model ID, for example claude-sonnet-4-6')
    ap.add_argument('--provider', choices=('anthropic', 'openai'), help='API provider (default: anthropic)')
    ap.add_argument('--base-url', help='Optional API endpoint, including OpenAI-compatible services')
    ap.add_argument('--json-mode', action='store_true', help='Request JSON without enforcing the schema at the API')
    ap.add_argument('--scene', type=Path, help='Scene record (JSON) from SIREN inference')
    ap.add_argument('--context',
                    type=Path,
                    help='JSON object with image measurements: image_width_px for half, '
                    'valid_area_px and pixel_size_m for count_density, pixel_size_m for distance')
    ap.add_argument('--output', type=Path, required=True, help='Output JSON file; must not exist')
    args = ap.parse_args()
    if args.context and not args.scene:
        ap.error('--context requires --scene')
    if args.output.exists():
        ap.error(f'{args.output} already exists; choose a new output file')
    if args.compiled:
        if args.model or args.provider or args.base_url or args.json_mode:
            ap.error('--compiled uses its saved settings; omit model/provider overrides')
        result = json.loads(args.compiled.read_text())
        if not isinstance(result, dict) or not {'program', 'reason'} <= set(result):
            ap.error(f'{args.compiled} needs "program" and "reason" fields')
    else:
        if not args.model:
            ap.error('--question requires --model')
        key = 'OPENAI_API_KEY' if args.provider == 'openai' else 'ANTHROPIC_API_KEY'
        if not os.environ.get(key):
            ap.error(f'Set {key} to compile a question (any value works for a local server '
                     'given with --base-url), or reuse a saved program with --compiled')
        result = compile_question(args.question, model=args.model,
                                  provider=args.provider or 'anthropic', base_url=args.base_url,
                                  structured=not args.json_mode)
    try:
        validate_response(json.dumps({k: result[k] for k in ('program', 'reason')}))
    except ValueError as exc:
        ap.error(f'Invalid program: {exc}')
    # Drop any answer saved with the program; this run computes its own.
    result.pop('answer', None)
    result.pop('execution_inputs', None)
    if args.scene and result['program'] is not None:
        context = json.loads(args.context.read_text()) if args.context else None
        if context is not None and not isinstance(context, dict):
            ap.error('--context must hold a JSON object')
        result['answer'] = execute_compiled(result, json.loads(args.scene.read_text()), context=context)
        result['execution_inputs'] = {
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (args.scene, args.context) if p is not None
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(
        json.dumps({
            'program': result['program'],
            'reason': result['reason'],
            'answer': result.get('answer')
        }))


if __name__ == '__main__':
    main()
