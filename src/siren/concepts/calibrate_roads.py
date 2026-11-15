"""Select road-access thresholds by validation macro-F1 and counting thresholds by
validation Count-MAE. Search grids and tie-breaking settings are in siren.settings."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from siren.files import read_unreadable_scenes
from siren.metrics import macro_f1
from siren.settings import (BLOCKED_COUNT_THRESHOLD, QUERY_THRESHOLDS, ROAD_MASK_COLUMNS,
                            ROAD_ACCESS_GRID as ACCESS_GRID, ROAD_COUNT_GRID as COUNT_GRID,
                            ROAD_COUNT_TIE_THRESHOLD)


def select_access(scores, labels):
    """Choose the road-access threshold from road-segment scores and labels (1 = blocked)."""
    (scores, labels) = (np.asarray(scores, float), np.asarray(labels))
    macro = [macro_f1(labels, (scores >= t).astype(int)) for t in ACCESS_GRID]
    i = int(np.argmax(macro))  # first maximum
    return {'threshold': float(ACCESS_GRID[i]), 'val_macro_f1': macro[i], 'n_segments': len(labels),
            'n_blocked': int((labels == 1).sum()), 'grid': ACCESS_GRID.tolist(),
            'macro_f1_by_threshold': macro}


def select_count(scenes):
    """Choose the counting threshold with the lowest Count-MAE over (road scores, reference count)
    pairs. Ties go to the threshold closest to ROAD_COUNT_TIE_THRESHOLD (0.45), then to the
    smaller one."""
    errors = [sum(abs(sum(s >= t for s in scores) - reference) for (scores, reference) in scenes)
              for t in COUNT_GRID]
    i = min(range(len(COUNT_GRID)),
            key=lambda j: (errors[j], round(abs(COUNT_GRID[j] - ROAD_COUNT_TIE_THRESHOLD), 12),
                           COUNT_GRID[j]))
    n = len(scenes)
    return {'threshold': COUNT_GRID[i], 'val_count_mae': errors[i] / n, 'n_val_scenes': n,
            'grid': COUNT_GRID, 'val_mae_by_threshold': [e / n for e in errors]}


def read_road_scores(path):
    """Read a JSON list of road segments with chip_id, entity_id, label and score."""
    segments = json.loads(Path(path).read_text())
    if not segments or len({(r['chip_id'], r['entity_id']) for r in segments}) != len(segments):
        raise ValueError('Road segments must be nonempty and unique')
    if any(r['label'] not in (0, 1) or not 0 <= r['score'] <= 1 for r in segments):
        raise ValueError('Road segment labels must be 0 or 1, and scores between 0 and 1')
    return ([r['score'] for r in segments], [r['label'] for r in segments])


def read_scenes(args):
    """Pair each eligible validation scene's detected road scores with its reference count."""
    with open(args.split_csv, newline='') as f:
        labelled = {r['chip_id'] for r in csv.DictReader(f) if r['split'] == 'val'
                    and any((r.get(c) or '').strip() for c in ROAD_MASK_COLUMNS)}
    excluded = read_unreadable_scenes(args.unreadable_scenes)
    with open(args.references, newline='') as f:
        references = {r['chip_id']: int(r['count']) for r in csv.DictReader(f)}
    scenes = []
    for chip in sorted((set(references) & labelled) - excluded):
        record = json.loads((args.entities_dir / f'{chip}.json').read_text())
        if record['chip_id'] != chip or record['split'] != 'val':
            raise ValueError(f'Entity file does not match validation scene {chip}')
        roads = record['entities']['roads']
        scenes.append(([r['attributes']['is_road_blocked_score'] for r in roads], references[chip]))
    if not scenes:
        raise ValueError('No eligible validation scenes with road labels')
    return scenes


def main():
    """Read validation predictions and save the selected road thresholds."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--road-scores',
                    type=Path,
                    help='Road access: JSON list of annotated validation road segments with chip_id, '
                    'entity_id, label (1 = blocked) and blocked-road score')
    ap.add_argument('--entities-dir',
                    type=Path,
                    help='SIREN entity files for the validation split (blocked-road counting)')
    ap.add_argument('--references', type=Path, help='Validation blocked_roads.csv')
    ap.add_argument('--split-csv', type=Path, help='Scene split (assets/metadata/split.csv)')
    ap.add_argument('--unreadable-scenes', type=Path,
                    help='Validation scenes to skip (assets/metadata/unreadable_scenes.json)')
    ap.add_argument('--output', type=Path, required=True, help='Output thresholds JSON')
    args = ap.parse_args()
    if not (args.road_scores or args.entities_dir):
        ap.error('Supply --road-scores, --entities-dir or both')
    if args.entities_dir and not (args.references and args.split_csv and args.unreadable_scenes):
        ap.error('--entities-dir needs --references, --split-csv and --unreadable-scenes')
    out = {'rule': __doc__}  # the module docstring states the selection rules
    if args.road_scores:
        out['road_access'] = select_access(*read_road_scores(args.road_scores))
        out['road_access']['paper_threshold'] = QUERY_THRESHOLDS['is_road_blocked']
    if args.entities_dir:
        out['road_counting'] = select_count(read_scenes(args))
        out['road_counting']['paper_threshold'] = BLOCKED_COUNT_THRESHOLD
    print(json.dumps({k: {f: v[f] for f in ('threshold', 'paper_threshold')}
                      for (k, v) in out.items() if k != 'rule'}, indent=1))
    args.output.write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
