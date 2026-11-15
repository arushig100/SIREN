"""Score scene counts against reference counts: Count-MAE per question and, when all eight
questions are given, the count-error reduction relative to predicting zero.

A prediction CSV with an event column (data/predictions/external.csv) is scored per event."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from siren.files import integer_count
from siren.programs.recipes import COUNT_QUERIES


def index_scenes(rows):
    """Index rows by scene ID, rejecting empty, duplicated or path-like IDs."""
    result = {}
    for row in rows:
        chip = row.get('chip_id')
        if (not isinstance(chip, str) or not chip or any(c.isspace() for c in chip)
                or Path(chip).name != chip or chip in ('.', '..') or chip in result):
            raise ValueError(f'Empty, duplicated or invalid scene ID: {chip!r}')
        result[chip] = row
    if not result:
        raise ValueError('Empty scene list')
    return result


def read_references(path):
    """Read reference counts by scene ID from a chip_id,count CSV."""
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        if 'count' not in (reader.fieldnames or ()):
            raise ValueError(f'{path} needs columns chip_id and count')
        rows = index_scenes(list(reader))
    return {chip: integer_count(row['count']) for chip, row in rows.items()}


def read_rows(path):
    """Read one CSV of predictions from several systems for paired scoring."""
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        missing = {'question', 'chip_id', 'system', 'reference', 'prediction'} - fields
        if missing:
            raise ValueError(f'{path} needs columns question, chip_id, system, reference and '
                             f'prediction; missing {sorted(missing)}')
        rows = list(reader)
    seen = set()
    for row in rows:
        key = (row.get('event'), row['question'], row['chip_id'], row['system'])
        if key in seen:
            raise ValueError(f'Duplicate prediction: {key}')
        seen.add(key)
        for field in ('reference', 'prediction'):
            try:
                row[field] = integer_count(row[field])
            except ValueError:
                raise ValueError(f'Invalid {field}: {key}') from None
    return rows


def score(rows):
    """Return Count-MAE per question and system, and the always-zero MAE.

    Every system must cover the same scenes with the same reference counts."""
    if not rows:
        raise ValueError('No predictions to score')
    groups = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        if row['chip_id'] in groups[row['question']][row['system']]:
            raise ValueError(f"Duplicate prediction: {row['question']}/{row['system']}/{row['chip_id']}")
        groups[row['question']][row['system']][row['chip_id']] = row
    result = {}
    for question, systems in groups.items():
        first_system = next(iter(systems.values()))
        chips = sorted(first_system)
        references = np.array([first_system[c]['reference'] for c in chips])
        metrics = {}
        for system, records in systems.items():
            if set(records) != set(chips):
                raise ValueError(f'Incomplete or extra scene coverage: {question}/{system}; every '
                                 'system needs the same scenes')
            if [records[c]['reference'] for c in chips] != references.tolist():
                raise ValueError(f'Reference disagreement: {question}/{system}')
            error = np.abs(np.array([records[c]['prediction'] for c in chips]) - references)
            metrics[system] = {'count_mae': float(error.mean())}
        # Predicting zero everywhere has an error equal to the mean reference count.
        result[question] = {
            'n': len(chips),
            'always_zero_mae': float(references.mean()),
            'systems': metrics
        }
    return result


def align_predictions(predictions, references, question, system):
    """Join count predictions to reference counts by scene ID, never by row position."""
    with Path(predictions).open(newline='') as stream:
        predictions_by_id = index_scenes(list(csv.DictReader(stream)))
    predicted = {chip: integer_count(row['prediction']) for chip, row in predictions_by_id.items()}
    reference = read_references(references)
    if predicted.keys() != reference.keys():
        raise ValueError('Predictions must cover the same scene IDs as the reference file')
    return [
        dict(question=question, system=system, chip_id=chip, reference=reference[chip],
             prediction=predicted[chip])
        for chip in sorted(reference)
    ]


def count_error_reduction_vs_zero(results):
    """Score per system: 100 x (1 - mean over the eight questions of Count-MAE / always-zero MAE).

    results is the output of score() for all eight questions. 100 is exact counting and 0 is no
    better than predicting zero; this is not classification accuracy."""
    if len(results) != len(COUNT_QUERIES):
        raise ValueError('Count-error reduction requires all eight questions')
    systems = set.intersection(*(set(r['systems']) for r in results.values()))
    missing = set.union(*(set(r['systems']) for r in results.values())) - systems
    if missing:
        raise ValueError(f'Systems missing some questions: {sorted(missing)}')
    if any(r['always_zero_mae'] <= 0 for r in results.values()):
        raise ValueError('Normalized error requires positive reference mean counts')
    return {
        s:
        float(100 * (1 - np.mean(
            [r['systems'][s]['count_mae'] / r['always_zero_mae'] for r in results.values()])))
        for s in sorted(systems)
    }


def main():
    """Align predictions with reference counts and write evaluation metrics."""
    ap = argparse.ArgumentParser(description=__doc__)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument('--predictions',
                        type=Path,
                        help='Prediction CSV: chip_id,prediction with --references; '
                        'question,chip_id,system,reference,prediction (and optionally event) '
                        'without')
    source.add_argument('--entities-dir',
                        type=Path,
                        help='Run a SIREN query on saved scene entities; rows are labelled SIREN')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--references',
                    type=Path,
                    help='Reference count CSV, such as data/counts/test/damaged_buildings.csv')
    ap.add_argument('--question',
                    help='Name stored with the scores; with --entities-dir, the query to run')
    ap.add_argument('--system', help='System name stored with the scores, such as SIREN '
                    '(not used with --entities-dir)')
    ap.add_argument('--scene-ids', type=Path,
                    help='With --entities-dir: score only these scenes (one ID per line), for '
                    'example after a partial export')
    args = ap.parse_args()
    if args.output.exists():
        ap.error(f'{args.output} already exists; choose a new output file')
    if args.scene_ids and not args.entities_dir:
        ap.error('--scene-ids is used with --entities-dir')
    if args.entities_dir:
        if not (args.references and args.question):
            ap.error('--entities-dir requires --references and --question')
        from siren.evaluation.external import count_scene
        references = read_references(args.references)
        if args.scene_ids:
            wanted = {line.strip() for line in args.scene_ids.read_text().splitlines()
                      if line.strip()}
            unknown = wanted - references.keys()
            if unknown:
                raise ValueError(f'{len(unknown)} scene IDs are not in the reference file, for '
                                 f'example {sorted(unknown)[:3]}')
            references = {chip: count for chip, count in references.items() if chip in wanted}
        rows = []
        for chip, reference in references.items():
            path = args.entities_dir / (chip + '.json')
            if not path.exists():
                ap.error(f'{path} is missing; to score a partial export, pass --scene-ids')
            scene = json.loads(path.read_text())
            if scene.get('chip_id') != chip:
                raise ValueError('Scene ID does not match the entity file')
            rows.append(
                dict(chip_id=chip,
                     question=args.question,
                     system='SIREN',
                     reference=reference,
                     prediction=count_scene(scene, args.question)))
    elif args.references and not (args.question and args.system):
        ap.error('--references requires --question and --system')
    else:
        rows = (align_predictions(args.predictions, args.references, args.question, args.system)
                if args.references else read_rows(args.predictions))
    if any(row.get('event') for row in rows):
        events = defaultdict(list)
        for row in rows:
            if not row.get('event'):
                raise ValueError('Every row needs an event when any row has one')
            events[row['event']].append(row)
        output = {'events': {event: score(event_rows) for event, event_rows in events.items()}}
    else:
        result = score(rows)
        output = {'questions': result}
        if set(result) == set(COUNT_QUERIES):
            output['count_error_reduction_vs_zero'] = count_error_reduction_vs_zero(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(output, stream, indent=2)


if __name__ == '__main__':
    main()
