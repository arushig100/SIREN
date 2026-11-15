"""Select damage and destruction thresholds by validation Count-MAE.
Apply the inference detection settings before comparing predicted and reference counts.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from siren.files import read_unreadable_scenes
from siren.segmentation.building.utils import merge_duplicate_boxes
from siren.settings import BUILDINGS, DAMAGE_THRESHOLD_GRID

# Candidate columns: view, detection score, box corners, mask area, truncated flag,
# damage probability and destruction probability.
SCORE_COLUMNS = {'damaged': 8, 'destroyed': 9}
COUNT_QUESTIONS = {'damaged': 'damaged_buildings', 'destroyed': 'destroyed_buildings'}


def merge_candidates(candidates, views, full_score, tile_score):
    """Filter candidate views and scores, then remove overlapping detections."""
    if not len(candidates):
        return candidates
    view = candidates[:, 0].astype(int)
    keep = np.isin(view, views) & np.where(
        view == 0, candidates[:, 1] >= full_score,
        (candidates[:, 1] >= tile_score) & (candidates[:, 7] == 0))
    candidates = candidates[keep]
    kept = merge_duplicate_boxes(candidates[:, 2:6], candidates[:, 1],
                                 iou=BUILDINGS.merge_iou, containment=BUILDINGS.merge_containment)
    return candidates[kept]


def select_candidates(candidates):
    """Apply the inference tiling rule and return detections plus the number of used tiles."""
    base = candidates[(candidates[:, 0] == 0) &
                      (candidates[:, 1] >= BUILDINGS.base_score)]
    if len(base) < BUILDINGS.tile_trigger:
        return merge_candidates(candidates, [0], BUILDINGS.base_score, BUILDINGS.base_score), 0
    centers_x = (base[:, 2] + base[:, 4]) / 2
    centers_y = (base[:, 3] + base[:, 5]) / 2
    selected = []
    for i, (x, y) in enumerate(BUILDINGS.tile_origins()):
        inside = ((centers_x >= x) & (centers_x < x + BUILDINGS.tile_size) &
                  (centers_y >= y) & (centers_y < y + BUILDINGS.tile_size))
        if inside.sum() >= BUILDINGS.tile_min_buildings:
            selected.append(BUILDINGS.tile_view_start + i)
    merged = merge_candidates(candidates, [0, *selected], BUILDINGS.full_score,
                              BUILDINGS.tile_score)
    return merged, len(selected)


def main():
    """Read validation candidates and references, search thresholds and save the results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', type=Path, required=True,
                        help='candidates_val folder written by train_damage')
    parser.add_argument('--counts-dir', type=Path, required=True,
                        help='Count references with val/damaged_buildings.csv and '
                        'val/destroyed_buildings.csv (data/counts)')
    parser.add_argument('--output', type=Path, required=True, help='Output thresholds JSON')
    parser.add_argument('--unreadable-scenes', type=Path,
                        help='Validation scenes to skip (assets/metadata/unreadable_scenes.json)')
    args = parser.parse_args()
    candidates = {}
    for path in sorted(args.candidates.glob('*.json')):
        record = json.loads(path.read_text())
        candidates[record['chip_id']] = np.asarray(record['cands'], float).reshape(-1, 10)
    references = {}
    for concept, question in COUNT_QUESTIONS.items():
        with (args.counts_dir / 'val' / f'{question}.csv').open() as stream:
            references[concept] = {r['chip_id']: int(r['count']) for r in csv.DictReader(stream)}
    excluded = frozenset()
    if args.unreadable_scenes:
        excluded = read_unreadable_scenes(args.unreadable_scenes)
        references = {q: {c: y for c, y in rows.items() if c not in excluded}
                      for q, rows in references.items()}
    scenes = sorted(references['damaged'])
    if not scenes or set(scenes) != set(references['destroyed']):
        raise ValueError('Damage and destruction references must cover the same validation scenes')
    missing = set(scenes) - candidates.keys()
    if missing:
        raise ValueError(f'Missing candidate predictions for {len(missing)} validation scenes, '
                         f'for example {sorted(missing)[:3]}; --unreadable-scenes skips '
                         'the unreadable scenes')
    results = [select_candidates(candidates[scene]) for scene in scenes]
    grid = DAMAGE_THRESHOLD_GRID
    output = {'rule': __doc__, 'candidates': str(args.candidates), 'n_val_scenes': len(scenes),
              'unreadable_scenes': sorted(excluded),
              'tile_passes': float(np.mean([n for _, n in results])),
              'config': dict(tile_trigger=BUILDINGS.tile_trigger,
                             tile_min_buildings=BUILDINGS.tile_min_buildings,
                             full_score=BUILDINGS.full_score, tile_score=BUILDINGS.tile_score),
              'grid': grid.tolist()}
    for concept, column in SCORE_COLUMNS.items():
        reference = np.array([references[concept][scene] for scene in scenes], float)
        predictions = np.array([[(rows[:, column] >= threshold).sum() for rows, _ in results]
                                for threshold in grid])
        errors = np.abs(predictions - reference).mean(axis=1)
        best = int(np.argmin(errors))  # Keep the first minimum.
        output[concept] = {'threshold': float(grid[best]), 'val_count_mae': float(errors[best]),
                           'val_mae_by_threshold': errors.tolist()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
