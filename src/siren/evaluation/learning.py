"""Learning curves: fixed encoder and entity extraction, with damage classifiers trained
on each subset. Cache test-entity features once, apply each run's classifiers, then score
its counts."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from shapely.strtree import STRtree

from siren.files import expand_data_path, require_file
from siren.programs.geometry import shape
from siren.settings import NEAR_DISTANCE_PX


def counts(buildings, damage_scores, destruction_scores, damage_threshold, destruction_threshold):
    """Count damaged buildings, and destroyed buildings with no intact building nearby.

    Nearby means within NEAR_DISTANCE_PX. Gives the same answers as the damaged_buildings and
    destroyed_buildings_without_intact_neighbor programs; this direct version builds each
    footprint once and uses a spatial index, which is much faster over every run and scene."""
    geom = [shape(b, 'footprint') for b in buildings]
    intact = [
        g for (g, d, x) in zip(geom, damage_scores, destruction_scores)
        if d < damage_threshold and x < destruction_threshold
    ]
    destroyed = [g for (g, x) in zip(geom, destruction_scores) if x >= destruction_threshold]
    if intact:
        tree = STRtree(intact)
        isolated_count = sum(
            not any(g.distance(intact[i]) <= NEAR_DISTANCE_PX
                    for i in tree.query(g.buffer(NEAR_DISTANCE_PX)))
            for g in destroyed)
    else:
        isolated_count = len(destroyed)
    return {
        'damaged_buildings': int(sum((d >= damage_threshold for d in damage_scores))),
        'destroyed_buildings_without_intact_neighbor': int(isolated_count)
    }


def cache_features(args):
    """Recompute the exported test buildings' paired features and check the exported scores."""
    import torch
    from PIL import Image
    from siren.files import read_unreadable_scenes
    from siren.inference.pipeline import check_hashes, load_assets, load_models
    from siren.perception.model import POOLED_DIM
    args.output.mkdir(parents=True, exist_ok=False)
    # The model and heads that wrote the export, checked against the hashes in the asset config.
    assets = load_assets(args.assets)
    check_hashes(assets, ('base', 'damaged', 'destroyed', 'split_csv', 'unreadable_scenes'))
    # Load the classifiers and decision thresholds used by the export.
    (building, _) = load_models(assets, args.device)
    thresholds = building.thresholds
    excluded = read_unreadable_scenes(assets['unreadable_scenes']['path'])
    rows = sorted((r for r in csv.DictReader(open(assets['split_csv']['path']))
                   if r['split'] == 'test' and r['chip_id'] not in excluded),
                  key=lambda r: r['chip_id'])
    exports = {}
    for r in rows:
        raw = (args.entities_dir / (r['chip_id'] + '.json')).read_bytes()
        d = json.loads(raw)
        if (d['chip_id'] != r['chip_id'] or d['export']['pipeline'] != 'siren'
                or d['export']['thresholds'] != thresholds):
            raise ValueError(
                f"{r['chip_id']}: export does not match the configured model and heads")
        exports[r['chip_id']] = (d['entities']['buildings'], hashlib.sha256(raw).hexdigest())
    total = sum((len(v[0]) for v in exports.values()))
    print('test scenes', len(rows), 'entities', total, flush=True)
    feats = np.lib.format.open_memmap(args.output / 'test_feats.npy.tmp',
                                      mode='w+',
                                      dtype=np.float32,
                                      shape=(total, 2, POOLED_DIM))

    def load(p):
        """Read an RGB image tensor on the selected device."""
        with Image.open(require_file(expand_data_path(p))) as im:
            return torch.from_numpy(np.array(im.convert('RGB'))).permute(
                2, 0, 1).float().div(255).to(args.device)

    (index, recomputed, exported) = ([], {q: [] for q in thresholds}, {q: [] for q in thresholds})
    k = 0
    t0 = time.time()
    with torch.inference_mode():
        for (i, r) in enumerate(rows, 1):
            (buildings, sha) = exports[r['chip_id']]
            n = len(buildings)
            index.append({
                'chip_id': r['chip_id'],
                'start': k,
                'n': n,
                'ids': [b['id'] for b in buildings],
                'export_sha256': sha
            })
            if n:
                (pre, post) = (load(r[c]) for c in ('pre_image_path', 'post_image_path'))
                # Same features as the export: full-image FPNs pooled at the saved boxes.
                (images, _) = building.det.transform([pre], None)
                fpn_pre = building.model.fpn(images.tensors)
                boxes = torch.tensor([b['geometry']['bbox_xyxy'] for b in buildings],
                                     device=args.device,
                                     dtype=torch.float32)
                paired_features = torch.stack((building.pooled(
                    fpn_pre, boxes), building.pooled(building.model.fpn(post[None]), boxes)), 1)
                feats[k:k + n] = paired_features.cpu().numpy()
                for q in thresholds:
                    recomputed[q].append(
                        building.heads[q](paired_features).softmax(-1)[:, 1].cpu().numpy())
                    exported[q].append(np.array([b['attributes'][q + '_score'] for b in buildings]))
                k += n
            if i % 200 == 0:
                print(i, len(rows), f'{time.time() - t0:.0f}s', flush=True)
    if k != total:
        raise ValueError('Feature count does not match the export')
    feats.flush()
    del feats
    (args.output / 'test_feats.npy.tmp').rename(args.output / 'test_feats.npy')
    (args.output / 'index.json').write_text(
        json.dumps({
            'entities_dir': str(args.entities_dir),
            'n_entities': total,
            'dtype': 'float32',
            'shape': [total, 2, POOLED_DIM],
            'chips': index
        }))
    check = {
        'gpu': torch.cuda.get_device_name(0) if args.device.startswith('cuda') else args.device,
        'n_entities': total,
        'n_scenes': len(rows)
    }
    passed = True
    for q in thresholds:
        (new, saved) = (np.concatenate(recomputed[q]) if recomputed[q] else np.zeros(0),
                        np.concatenate(exported[q]) if exported[q] else np.zeros(0))
        flips = int(((new >= thresholds[q]) != (saved >= thresholds[q])).sum())
        max_diff = float(np.abs(new - saved).max()) if len(new) else 0.0
        counts_new = [int((np.asarray(x) >= thresholds[q]).sum()) for x in recomputed[q]]
        counts_saved = [int((np.asarray(x) >= thresholds[q]).sum()) for x in exported[q]]
        check[q] = {
            'max_abs_diff': max_diff,
            'mean_abs_diff': float(np.abs(new - saved).mean()) if len(new) else 0.0,
            'threshold_flips': flips,
            'scenes_with_count_change': int(sum(x != y for (x, y) in zip(counts_new, counts_saved))),
            'sum_abs_count_change': int(sum(abs(x - y) for (x, y) in zip(counts_new, counts_saved)))
        }
        # Accept small numerical differences, such as from another GPU: at most 0.02 in any score,
        # and threshold flips for at most 0.2% of entities.
        passed &= max_diff <= 0.02 and flips <= 0.002 * max(1, total)
    check['passed'] = bool(passed)
    (args.output / 'score_check.json').write_text(json.dumps(check, indent=1))
    print(json.dumps(check, indent=1), flush=True)
    if not passed:
        raise RuntimeError('Recomputed scores differ from the exported scores')
    (args.output / 'score_check_passed').write_text('ok\n')


def apply_heads(args):
    """Score the cached test features with one subset-trained pair of damage classifiers."""
    import torch
    from siren.inference.buildings import load_head
    if args.output.exists():
        raise FileExistsError(f'{args.output} already exists; choose a new output file')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not (args.cache / 'score_check_passed').is_file():
        raise ValueError('Feature cache has no score_check_passed file; rerun the cache command '
                         'and check score_check.json')
    idx = json.loads((args.cache / 'index.json').read_text())
    features = np.load(args.cache / 'test_feats.npy', mmap_mode='r')
    if list(features.shape) != idx['shape'] or features.dtype != np.float32:
        raise ValueError('Feature cache does not match index.json')
    out = {
        'chip_ids': np.array([c['chip_id'] for c in idx['chips']]),
        'starts': np.array([c['start'] for c in idx['chips']]),
        'counts': np.array([c['n'] for c in idx['chips']]),
        'export_sha256': np.array([c['export_sha256'] for c in idx['chips']])
    }
    if int(out['counts'].sum()) != features.shape[0]:
        raise ValueError('Feature cache does not match index.json')
    meta = {'cache': str(args.cache), 'n_entities': int(features.shape[0]), 'heads': {}}
    with torch.inference_mode():
        for q in ('is_damaged', 'is_destroyed'):
            f = args.heads / f'{q.removeprefix("is_")}.pth'
            meta['heads'][q] = {
                'file': str(f),
                'sha256': hashlib.sha256(f.read_bytes()).hexdigest()
            }
            net = load_head(f, args.device)
            sc = []
            for i in range(0, features.shape[0], 4096):
                sc.append(
                    net(torch.from_numpy(np.array(features[i:i + 4096])).to(
                        args.device)).softmax(-1)[:, 1].cpu().numpy())
            out[q] = (np.concatenate(sc) if sc else np.zeros(0)).astype(np.float32)
            print(q,
                  'scored',
                  len(out[q]),
                  'mean',
                  float(out[q].mean()) if len(out[q]) else None,
                  flush=True)
    np.savez_compressed(args.output, **out)
    args.output.with_suffix('.json').write_text(json.dumps(meta, indent=1))
    print('wrote', args.output, flush=True)


def score_heads(args):
    """Count with one run's saved scores and thresholds, then score against the reference counts."""
    references = {q: args.counts_dir / 'test' / f'{q}.csv'
                  for q in ('damaged_buildings', 'destroyed_buildings_without_intact_neighbor')}
    chips = sorted((r['chip_id'] for r in csv.DictReader(open(references['damaged_buildings']))))
    saved_scores = np.load(args.run_dir / 'test_scores.npz')
    thresholds = json.loads((args.run_dir / 'thresholds.json').read_text())
    (damage_threshold, destruction_threshold) = (thresholds['damaged']['threshold'],
                                                 thresholds['destroyed']['threshold'])
    pos = {
        c: (int(s), int(n), h)
        for (c, s, n, h) in zip(saved_scores['chip_ids'], saved_scores['starts'],
                                saved_scores['counts'], saved_scores['export_sha256'])
    }
    predictions = {}
    for c in chips:
        raw = (args.entities_dir / f'{c}.json').read_bytes()
        buildings = json.loads(raw)['entities']['buildings']
        (s, n, h) = pos[c]
        if n != len(buildings) or h != hashlib.sha256(raw).hexdigest():
            raise ValueError(f'Export changed since the cache was built: {c}')
        (damage_scores, destruction_scores) = (saved_scores['is_damaged'][s:s + n].tolist(),
                                               saved_scores['is_destroyed'][s:s + n].tolist())
        predictions[c] = counts(buildings, damage_scores, destruction_scores, damage_threshold,
                                destruction_threshold)
    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    for q in references:
        with open(out_dir / f'predictions_test_{q}.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(('chip_id', 'prediction'))
            w.writerows(((c, predictions[c][q]) for c in chips))
    print('thresholds', damage_threshold, destruction_threshold, 'scenes', len(chips),
          'predictions written to', out_dir)
    res = {
        'run_dir': str(args.run_dir),
        'thresholds': {
            'damaged': damage_threshold,
            'destroyed': destruction_threshold
        },
        'n': len(chips)
    }
    for q in references:
        reference = {r['chip_id']: int(r['count']) for r in csv.DictReader(open(references[q]))}
        if set(reference) != set(chips):
            raise ValueError(f'{references[q]} lists different scenes from the damaged-building '
                             'reference file')
        y = np.array([reference[c] for c in chips], float)
        p = np.array([predictions[c][q] for c in chips], float)
        res[q] = {'count_mae': float(np.abs(p - y).mean())}
    print(json.dumps(res, indent=1))
    (out_dir / 'test_results.json').write_text(json.dumps(res, indent=1) + '\n')


def main():
    """Cache features, apply trained classifiers or score a learning-curve run."""
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    cache = sub.add_parser('cache',
                           help='Recompute and check test-entity features from a SIREN export')
    cache.add_argument('--assets',
                       type=Path,
                       required=True,
                       help='Asset config, such as configs/assets.json')
    for name in ('entities-dir', 'output'):
        cache.add_argument('--' + name, type=Path, required=True)
    cache.add_argument('--device', default=None,
                       help='torch device (default: cuda if available, else cpu)')
    cache.set_defaults(run=cache_features)
    apply = sub.add_parser('apply', help="Score cached features with one run's damage classifiers")
    for name in ('cache', 'heads', 'output'):
        apply.add_argument('--' + name, type=Path, required=True)
    apply.add_argument('--device', default=None,
                       help='torch device (default: cuda if available, else cpu)')
    apply.set_defaults(run=apply_heads)
    score = sub.add_parser('score', help='Count and score the two learning-curve queries')
    for name in ('run-dir', 'entities-dir', 'counts-dir', 'output'):
        score.add_argument('--' + name, type=Path, required=True)
    score.set_defaults(run=score_heads)
    args = ap.parse_args()
    if getattr(args, 'device', '') is None:
        import torch
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.run(args)


if __name__ == '__main__':
    main()
