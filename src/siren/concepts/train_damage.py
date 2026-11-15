"""Train the damage and destruction classifiers on frozen paired entity features.
--train-chips restricts training to a shared scene list, as in the learning curves."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from siren.files import expand_data_path, read_unreadable_scenes, require_file
from siren.metrics import macro_f1


def parse_args():
    """Read the model paths, training subset and classifier settings."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True, help='Output folder')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--seeds', default='0',
                    help='Comma-separated training seeds; each writes a seed_<n> folder')
    ap.add_argument('--train-chips', type=Path,
                    help='Training scene IDs, one per line (for example from data/subsets)')
    ap.add_argument('--split-csv', type=Path, required=True,
                    help='Scene split with image paths (assets/metadata/split.csv)')
    ap.add_argument('--unreadable-scenes', type=Path, required=True,
                    help='Validation scenes to skip (assets/metadata/unreadable_scenes.json)')
    ap.add_argument('--base-checkpoint', type=Path, required=True,
                    help='base.pth, which supplies the frozen features')
    ap.add_argument('--labels-dir', type=Path, required=True,
                    help='Per-entity label folder (data/labels)')
    ap.add_argument('--buildings-csv', type=Path, required=True,
                    help='Reference building boxes (data/buildings.csv)')
    ap.add_argument('--device', default=None,
                    help='torch device (default: cuda if available, else cpu)')
    args = ap.parse_args()
    args.device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    return args


def load_splits(args):
    """Load the training and validation scenes and apply the optional shared training subset."""
    excluded = read_unreadable_scenes(args.unreadable_scenes)
    with open(args.split_csv, newline='') as f:
        rows = list(csv.DictReader(f))
    splits = {
        s: sorted((r for r in rows if r['split'] == s and r['chip_id'] not in excluded),
                  key=lambda r: r['chip_id'])
        for s in ('train', 'val')
    }
    subset_info = None
    if args.train_chips:
        keep = [line.strip() for line in args.train_chips.read_text().splitlines() if line.strip()]
        if len(keep) != len(set(keep)):
            raise ValueError('Duplicate scene IDs in --train-chips')
        missing = sorted(set(keep) - {r['chip_id'] for r in splits['train']})
        if missing:
            raise ValueError(f'Scenes not in the readable train split: {missing[:5]}')
        splits['train'] = [r for r in splits['train'] if r['chip_id'] in set(keep)]
        if len(splits['train']) != len(keep):
            raise ValueError('--split-csv lists a training scene more than once')
        subset_info = {
            'file': str(args.train_chips),
            'n': len(keep),
            'sha256_sorted_ids': hashlib.sha256('\n'.join(sorted(keep)).encode()).hexdigest()
        }
    return (splits, subset_info)


# Training detections: a scene whose full view has at least this many detections scoring at
# least BUILDINGS.base_score is also run on every tile. This pool is broader than the inference
# rule (BUILDINGS.tile_trigger detections, tiles with BUILDINGS.tile_min_buildings centers),
# which calibrate_damage applies to these candidates.
TRAINING_TILE_TRIGGER = 30


def box_iou(a, b):
    """Pairwise IoU of two box tensors: (len(a), len(b))."""
    width = (torch.minimum(a[:, None, 2], b[None, :, 2]) -
             torch.maximum(a[:, None, 0], b[None, :, 0])).clamp(min=0)
    height = (torch.minimum(a[:, None, 3], b[None, :, 3]) -
              torch.maximum(a[:, None, 1], b[None, :, 1])).clamp(min=0)
    overlap = width * height
    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None]
    return overlap / (area_a + area_b - overlap + 1e-09)


def extract_features(args, splits):
    """Cache paired building features and their damage labels for both splits."""
    from siren.concepts.registry import get_concepts
    from siren.inference.buildings import detect_buildings, pool_building_features, upsampled_tile
    from siren.perception.data import load_concept_supervision
    from siren.perception.loading import load_checkpoint
    from siren.perception.model import context_pad
    from siren.settings import BUILDINGS, ROADS
    # The damage heads read features pooled with the classifiers' default context margin.
    if context_pad('is_damaged') != (BUILDINGS.context_pad, 0.0):
        raise ValueError('Damage features must use the default context margin')
    # Label columns 0 and 1 of every feature row; calibrate_damage selects the thresholds.
    concept_names = ['is_damaged', 'is_destroyed']
    model = load_checkpoint(args.base_checkpoint, extraction=ROADS, device=args.device).model
    device = args.device
    references = {}
    for split in ('train', 'val'):
        supervision = load_concept_supervision(labels_dir=args.labels_dir,
                                               split_csv=args.split_csv,
                                               buildings_csv=args.buildings_csv,
                                               split=split,
                                               concepts=get_concepts(concept_names))
        for (column, name) in enumerate(concept_names):
            for r in supervision[name]:
                references.setdefault(r.chip_id,
                                      {}).setdefault(r.entity_id,
                                                     [r.bbox, -1, -1])[1 + column] = r.label
    if not references:
        raise ValueError(f'No damage labels found under {args.labels_dir}/per_entity')

    def load(path):
        """Read an image as an RGB tensor on the selected device."""
        with Image.open(require_file(expand_data_path(path))) as im:
            if im.size != (BUILDINGS.image_size, BUILDINGS.image_size):
                raise ValueError(f'Expected a 1024x1024 image: {path}')
            return torch.from_numpy(np.array(im.convert('RGB'))).permute(
                2, 0, 1).float().div(255).to(device)

    def detect(image, x0, y0, size, view):
        """Detections scoring >= BUILDINGS.detector_score_floor in one view, as candidate rows:
        view, detection score, x0, y0, x1, y1, mask area, truncated flag (always 0, because
        truncated masks are dropped)."""
        fpn, found = detect_buildings(model, image, x0, y0, size, view,
                                      BUILDINGS.detector_score_floor)
        return fpn, [[d['view'], d['score'], *d['box'], d['area'], 0] for d in found]

    @torch.inference_mode()
    def scene(row):
        """Candidates, paired features (N, 2, 6272), labels and reference-box flags of a scene."""
        pre, post = (load(row[k]) for k in ('pre_image_path', 'post_image_path'))
        fpn_pre, candidates = detect(pre, 0, 0, BUILDINGS.image_size, 0)
        fpn_post = model.fpn(post[None])
        if sum(c[1] >= BUILDINGS.base_score for c in candidates) >= TRAINING_TILE_TRIGGER:
            for (index, (x0, y0)) in enumerate(BUILDINGS.tile_origins()):
                tile = upsampled_tile(pre, x0, y0, BUILDINGS.tile_size)
                candidates += detect(tile, x0, y0, BUILDINGS.tile_size,
                                     BUILDINGS.tile_view_start + index)[1]
        boxes_and_labels = list(references.get(row['chip_id'], {}).values())
        boxes = torch.tensor([c[2:6] for c in candidates] + [r[0] for r in boxes_and_labels],
                             device=device,
                             dtype=torch.float32).reshape(-1, 4)
        if not len(boxes):
            return (candidates, np.zeros((0, 2, 6272), np.float16), np.zeros((0, 2), np.int8),
                    np.zeros(0, bool))
        features = torch.stack((pool_building_features(fpn_pre, boxes),
                                pool_building_features(fpn_post, boxes)), 1)
        # Uncertain matches keep label -1 and are excluded from training. A detection takes the
        # labels of its highest-IoU reference box when IoU >= 0.5 and is negative below IoU 0.2;
        # in a scene with no labelled building every detection is negative.
        n = len(candidates)
        labels = -np.ones((len(boxes), 2), np.int8)
        if boxes_and_labels:
            labels[n:] = np.array([r[1:] for r in boxes_and_labels], np.int8)
            if candidates:
                best_iou, best_reference = box_iou(boxes[:n], boxes[n:]).max(1)
                matched = labels[n:][best_reference.cpu().numpy()]
                labels[:n] = np.where((best_iou >= 0.5).cpu().numpy()[:, None], matched,
                                      np.where((best_iou < 0.2).cpu().numpy()[:, None], 0, -1))
        elif candidates:
            labels[:n] = 0
        return (candidates, features.half().cpu().numpy(), labels, np.arange(len(boxes)) >= n)

    features_by_split = {}
    for split in ('train', 'val'):
        start = time.time()
        # is_reference marks annotated boxes, as opposed to detector proposals.
        features, labels, is_reference, scenes = [], [], [], []
        for (i, row) in enumerate(splits[split], 1):
            try:
                (candidates, f, label, reference) = scene(row)
            except Exception as ex:
                raise RuntimeError(f"Cannot extract {split} scene {row['chip_id']}: {ex}") from ex
            features.append(f)
            labels.append(label)
            is_reference.append(reference)
            scenes.append((row['chip_id'], row.get('disaster_id', ''), candidates))
            if i % 200 == 0:
                print(split, i, len(splits[split]), f'{time.time() - start:.0f}s', flush=True)
        if not features:
            raise ValueError(f'No scenes yielded features for the {split} split')
        features_by_split[split] = dict(features=np.concatenate(features),
                                        labels=np.concatenate(labels),
                                        is_reference=np.concatenate(is_reference),
                                        scenes=scenes)
        del features
        d = features_by_split[split]
        print(split, 'entities', len(d['labels']), 'reference boxes', int(d['is_reference'].sum()),
              'detections', int((~d['is_reference']).sum()), 'labelled detections',
              int((~d['is_reference'] & (d['labels'][:, 0] >= 0)).sum()), flush=True)
    return (features_by_split, references, concept_names, device)


def train_heads(args, features_by_split, references, concept_names, device, subset_info):
    """Train each damage classifier, save its best validation checkpoint and score the
    validation candidates."""
    from siren.inference.buildings import BuildingConceptHead

    class TrainingHead(BuildingConceptHead):
        """Augment both dates together before applying the damage classifier."""

        def forward(self, x):
            """In training, apply one random flip/transpose per batch to both dates' 7x7 grids."""
            if self.training:
                x = x.reshape(-1, 2, 128, 7, 7)
                if torch.rand(1) < 0.5:
                    x = x.flip(3)
                if torch.rand(1) < 0.5:
                    x = x.flip(4)
                if torch.rand(1) < 0.5:
                    x = x.transpose(3, 4)
                x = x.reshape(-1, 2, 6272)
            return super().forward(x)

    def predict(net, features):
        """Evaluate a classifier on batches of cached features."""
        net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(features), 4096):
                batch = torch.from_numpy(features[i:i + 4096]).float().to(device)
                out.append(net(batch).softmax(-1)[:, 1].cpu())
        return torch.cat(out).numpy() if out else np.zeros(0)

    # Each seed writes its classifiers and validation candidates to its own seed_<n> folder.
    seeds = [int(value) for value in args.seeds.split(',')]
    training_log = {}
    train, val = features_by_split['train'], features_by_split['val']
    for seed in seeds:
        run_dir = args.output / f'seed_{seed}'
        run_dir.mkdir(exist_ok=True)
        val_scores = {}
        for (column, q) in enumerate(concept_names):
            torch.manual_seed(seed)
            np.random.seed(seed)
            # Every labelled training row: reference boxes and matched detections alike.
            idx = np.nonzero(train['labels'][:, column] >= 0)[0]
            y = torch.from_numpy(train['labels'][idx, column].astype(np.int64))
            train_features = torch.from_numpy(train['features'])
            vidx = np.nonzero(val['labels'][:, column] >= 0)[0]
            yv = val['labels'][vidx, column]
            val_features = val['features'][vidx]
            val_reference = val['is_reference'][vidx]
            counts = np.bincount(y.numpy(), minlength=2)
            if np.any(counts == 0):
                raise ValueError(f'{q}: training requires both classes; got {counts.tolist()}')
            if not len(vidx):
                raise ValueError(f'{q}: no labeled validation entities')
            # Inverse-square-root class weights.
            w = torch.tensor((counts.sum() / counts)**0.5, dtype=torch.float32, device=device)
            net = TrainingHead().to(device)
            opt = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.0001)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
            best = (-1, None, -1)
            for ep in range(args.epochs):
                net.train()
                perm = torch.randperm(len(idx))
                for i in range(0, len(perm), 512):
                    b = perm[i:i + 512]
                    opt.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(net(train_features[idx[b.numpy()]].float().to(device)),
                                           y[b].to(device),
                                           weight=w)
                    loss.backward()
                    opt.step()
                sch.step()
                pv = predict(net, val_features)
                f1_all = macro_f1(yv, (pv >= 0.5).astype(int))
                f1_reference = macro_f1(yv[val_reference], (pv[val_reference] >= 0.5).astype(int))
                print(f'seed_{seed}', q, 'ep', ep, 'n_train', len(idx),
                      f'val macroF1 all {f1_all:.4f} reference boxes {f1_reference:.4f}',
                      flush=True)
                # Keep the epoch with the best validation macro-F1 at 0.5 on all labelled rows.
                if f1_all > best[0]:
                    best = (f1_all, {
                        k: v.detach().cpu().clone()
                        for (k, v) in net.state_dict().items()
                    }, ep)
            net.load_state_dict(best[1])
            torch.save({'model_state_dict': best[1]}, run_dir / f'{q.removeprefix("is_")}.pth')
            training_log[f'seed_{seed}/{q}'] = {
                'val_macro_f1_all': best[0],
                'epoch': best[2],
                'n_train': len(idx),
                'class_counts': counts.tolist()
            }
            val_scores[q] = predict(net, val['features'])
        out_dir = run_dir / 'candidates_val'
        out_dir.mkdir(exist_ok=True)
        k = 0
        for (chip, disaster, candidates) in val['scenes']:
            rows = []
            for (j, candidate) in enumerate(candidates):
                # Columns 8 and 9: damage and destruction scores (read by calibrate_damage).
                rows.append(candidate + [
                    round(float(val_scores['is_damaged'][k + j]), 5),
                    round(float(val_scores['is_destroyed'][k + j]), 5)
                ])
            # Feature rows also hold the scene's reference boxes after its candidates.
            k += len(candidates) + len(references.get(chip, {}))
            (out_dir / (chip + '.json')).write_text(
                json.dumps({
                    'chip_id': chip,
                    'disaster_id': disaster,
                    'cands': rows
                }))
        if k != len(val['labels']):
            raise ValueError(f'Feature rows ({len(val["labels"])}) do not match the candidates '
                             f'and reference boxes ({k})')
        training_log['_subset'] = subset_info
        training_log['_n_train_scenes_used'] = len(train['scenes'])
        (args.output / 'train_log.json').write_text(json.dumps(training_log, indent=1))


def main():
    """Extract paired features and train the damage and destruction classifiers."""
    args = parse_args()
    (splits, subset_info) = load_splits(args)
    args.output.mkdir(parents=True, exist_ok=True)
    (features_by_split, references, concept_names, device) = extract_features(args, splits)
    train_heads(args, features_by_split, references, concept_names, device, subset_info)


if __name__ == '__main__':
    main()
