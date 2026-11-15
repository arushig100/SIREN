"""Train paired-image count baselines with validation checkpoint selection."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from siren.baselines.data import (ChipCountDataset, collate_chips, find_unreadable_images,
                                  load_split_index)
from siren.baselines.encoders import ENCODERS, build_encoder
from siren.baselines.model import ChipCountModel
from siren.files import read_unreadable_scenes
from siren.metrics import count_mae, round_count
from siren.programs.recipes import COUNT_QUERIES
from siren.settings import IMAGE_SIZE

# Paper recipe for each encoder; the effective batch is batch_size x accum = 32.
RECIPES = {
    'satlas': dict(lr=1e-4, batch_size=2, accum=16),
    'dinov3': dict(lr=1e-5, batch_size=1, accum=32),
    'remoteclip': dict(lr=1e-5, batch_size=2, accum=16),
    'skyclip': dict(lr=1e-5, batch_size=2, accum=16),
}
# Questions about roads use only scenes with a road mask in the split file.
ROAD_QUESTIONS = ('blocked_roads', 'damaged_buildings_without_clear_road')


def train_one_epoch(model, loader, optim, sched, device, *, amp, accum) -> float:
    """One epoch of L1 count regression; returns the mean training absolute error."""
    model.train()
    tot, n = 0.0, 0
    optim.zero_grad(set_to_none=True)
    n_batches = len(loader)
    for bi, b in enumerate(loader, 1):
        pre = b['pre'].to(device, non_blocking=True)
        post = b['post'].to(device, non_blocking=True)
        tgt = b['count'].to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
            pred = model(pre, post)
        # Train on continuous counts; rounding is applied only for evaluation.
        loss = F.l1_loss(pred.float(), tgt)
        (loss / accum).backward()
        # Update after every accum minibatches and at the end of the epoch.
        if bi % accum == 0 or bi == n_batches:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
            optim.step()
            optim.zero_grad(set_to_none=True)
        tot += float(loss.detach()) * pre.size(0)
        n += pre.size(0)
    sched.step()  # cosine schedule over epochs
    return tot / max(1, n)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    """Predict every scene; counts are the raw outputs rounded to non-negative integers."""
    model.eval()
    raw, references, cids, dids = [], [], [], []
    for b in loader:
        pre = b['pre'].to(device, non_blocking=True)
        post = b['post'].to(device, non_blocking=True)
        p = model(pre, post).cpu().numpy().reshape(-1)
        raw.extend(p.tolist())
        references.extend(b['count'].numpy().tolist())
        cids.extend(b['chip_id'])
        dids.extend(b['disaster_id'])
    if not all(math.isfinite(r) for r in raw):
        raise FloatingPointError('The model predicted a non-finite count')
    preds = [round_count(r) for r in raw]
    return {
        'mae': count_mae(references, preds),
        'n': len(references),
        'preds': preds,
        'references': references,
        'chip_ids': cids,
        'disaster_ids': dids,
        'raw': raw,
    }


def per_disaster(references, preds, dids) -> dict:
    """Count-MAE for each disaster."""
    by = defaultdict(lambda: ([], []))
    for g, p, d in zip(references, preds, dids):
        by[d][0].append(g)
        by[d][1].append(p)
    return {d: {'count_mae': count_mae(g, p), 'n': len(g)} for d, (g, p) in by.items()}


def fit(model, loaders, *, encoder, question, epochs, lr, weight_decay, device, out_dir,
        amp, accum, run_info=None) -> dict:
    """Train, keep the epoch with the lowest validation Count-MAE (best.pth) and score it on test."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    best = {'val_mae': float('inf')}
    log = out_dir / 'train_log.jsonl'
    log.write_text('')
    for ep in range(1, epochs + 1):
        tr = train_one_epoch(model, loaders['train'], optim, sched, device, amp=amp, accum=accum)
        va = evaluate(model, loaders['val'], device)
        print(f"[ep{ep:3d}] train_mae={tr:.3f} | val_mae={va['mae']:.3f}", flush=True)
        with log.open('a') as f:
            f.write(json.dumps({'epoch': ep, 'train_mae': tr, 'val_mae': va['mae']}) + '\n')
        if va['mae'] < best['val_mae']:
            best = {'epoch': ep, 'val_mae': va['mae'], 'val': va}
            torch.save({'model': model.state_dict(), 'epoch': ep, 'encoder': encoder,
                        'question': question}, out_dir / 'best.pth')
    if 'epoch' not in best:
        raise RuntimeError('No epoch gave a finite validation Count-MAE')
    checkpoint = torch.load(out_dir / 'best.pth', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'], strict=True)
    te = evaluate(model, loaders['test'], device)
    va = best['val']
    results = {
        'encoder': encoder,
        'question': question,
        'best_epoch': best['epoch'],
        'selection': 'val_mae',
        'overall': {
            'count_mae': te['mae'],
            'n': te['n']
        },
        'val': {
            'count_mae': va['mae'],
        },
        'per_disaster': per_disaster(te['references'], te['preds'], te['disaster_ids'])
    }
    if run_info:
        results['run_info'] = run_info
    (out_dir / 'results.json').write_text(json.dumps(results, indent=2))
    # Rounded predictions and raw model outputs of the selected checkpoint.
    for name, ev in (('val', va), ('test', te)):
        with (out_dir / f'predictions_{name}.csv').open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['chip_id', 'disaster_id', 'reference', 'prediction', 'raw'])
            for c, d, g, p, r in zip(ev['chip_ids'], ev['disaster_ids'], ev['references'],
                                       ev['preds'], ev['raw']):
                w.writerow([c, d, int(g), p, r])
    print(f"\n[done] {encoder} {question} best_epoch={best['epoch']} TEST mae={te['mae']:.3f}",
          flush=True)
    return results


class PairedFlip(torch.utils.data.Dataset):
    """Flip both images horizontally together so they stay aligned."""

    def __init__(self, base, p: float = 0.5):
        """Wrap a dataset with the same random flip for both images."""
        self.base, self.p = base, p

    def __len__(self):
        """Return the number of scenes in the wrapped dataset."""
        return len(self.base)

    def __getitem__(self, i):
        """Apply a shared random flip to the paired images."""
        item = self.base[i]
        if torch.rand(()) < self.p:
            item['pre'] = torch.flip(item['pre'], dims=[-1])
            item['post'] = torch.flip(item['post'], dims=[-1])
        return item


def parse_args():
    """Read the encoder, question, input paths and training settings."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--encoder', required=True, choices=sorted(ENCODERS))
    ap.add_argument('--question',
                    required=True,
                    choices=COUNT_QUERIES,
                    help='Count question to train; --counts-dir holds its train, val and test CSVs')
    ap.add_argument(
        '--ckpt',
        default=None,
        help='Pretrained encoder weights: required for remoteclip and skyclip, optional for '
        'dinov3 (otherwise loaded by timm), unused for satlas')
    ap.add_argument('--split-csv', required=True,
                    help='assets/metadata/split.csv: image paths and road masks of every scene')
    ap.add_argument('--unreadable-scenes',
                    default=None,
                    help='assets/metadata/unreadable_scenes.json: validation scenes to skip '
                    '(required with the released data, which has one empty image)')
    ap.add_argument('--counts-dir', required=True,
                    help='Folder with train/, val/ and test/ reference count CSVs, one per question')
    ap.add_argument('--output', required=True,
                    help='Output folder for the checkpoint, predictions and results')
    ap.add_argument('--train-subset',
                    type=int,
                    help='Train on the first N scenes of a seeded random order (nested subsets)')
    ap.add_argument('--subset-seed',
                    type=int,
                    default=0,
                    help='Seed of that order. For the damaged, destroyed, large destroyed and isolated '
                    'destroyed questions, seeds 0 and 1 reproduce data/subsets, and seed 2 the 25- '
                    'and 100-scene lists')
    ap.add_argument('--train-chips',
                    type=Path,
                    help='File of training scene IDs, one per line (for example from data/subsets); '
                    'each epoch keeps the full-data number of samples')
    ap.add_argument('--activation-checkpointing',
                    action=argparse.BooleanOptionalAction,
                    default=None,
                    help='Recompute activations to save GPU memory (default: on for the ViT encoders)')
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--batch-size', type=int, help='Default: the encoder\'s paper recipe')
    ap.add_argument('--lr', type=float, help='Default: the encoder\'s paper recipe')
    ap.add_argument('--weight-decay', type=float, default=0.0001)
    ap.add_argument('--num-workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--accum',
                    type=int,
                    help='Gradient accumulation steps; effective batch = batch-size * accum. '
                    'Default: the encoder\'s paper recipe')
    ap.add_argument('--amp',
                    action=argparse.BooleanOptionalAction,
                    default=True,
                    help='Use bfloat16 autocast in training steps (paper setting)')
    args = ap.parse_args()
    # Settings left unset follow the encoder's paper recipe.
    for name, value in RECIPES[args.encoder].items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    # Input size: 1022 (73 patches of 14 pixels) for the CLIP encoders, 1024 for the others.
    args.image_size = 1022 if args.encoder in ('remoteclip', 'skyclip') else IMAGE_SIZE
    if args.activation_checkpointing is None:
        args.activation_checkpointing = args.encoder != 'satlas'
    return ap, args


def build_loaders(args, encoder, counts_dir, split_index, excluded, run_info, device):
    """Build scene loaders, keeping the full-data step budget for learning curves."""
    loaders = {}
    tx = encoder.transform(args.image_size)
    for s in ('train', 'val', 'test'):
        reference = counts_dir / s / f'{args.question}.csv'
        ds = ChipCountDataset(reference, split_index, tx, excluded_chip_ids=excluded)
        if args.question in ROAD_QUESTIONS:
            no_mask = [r['chip_id'] for r in ds.rows if not split_index[r['chip_id']]['has_road_mask']]
            if no_mask:
                raise ValueError(f'{s}: {len(no_mask)} {args.question} reference scenes have no '
                                 f'road mask in the split file, for example {no_mask[:3]}')
        full_n = len(ds)
        if s == 'train' and (args.train_subset is not None or args.train_chips):
            if args.train_chips:
                ids = args.train_chips.read_text().splitlines()
                if not ids:
                    raise ValueError('Training scene list is empty')
                if len(ids) != len(set(ids)) or not set(ids) <= {r['chip_id'] for r in ds.rows}:
                    raise ValueError('Training list contains duplicate or non-training scenes')
                keep_ids = set(ids)
                ds.rows = [r for r in ds.rows if r['chip_id'] in keep_ids]
            else:
                if not 1 <= args.train_subset <= full_n:
                    raise ValueError('Training subset size outside the available training set')
                # Offset 1000 reproduces the lists in data/subsets for questions whose training
                # count file lists all 4,627 training scenes.
                perm = np.random.RandomState(1000 + args.subset_seed).permutation(full_n)
                keep = sorted(perm[:args.train_subset].tolist())
                ds.rows = [ds.rows[i] for i in keep]
            run_info['train_subset'] = dict(n=len(ds.rows),
                                            subset_seed=args.subset_seed,
                                            chip_ids=[r['chip_id'] for r in ds.rows],
                                            steps_budget_scenes_per_epoch=full_n)
        print(f'[data] {s}: n={len(ds)}', flush=True)
        sampler = None
        if s == 'train':
            counts = np.array([r['count'] for r in ds.rows], dtype=np.float64)
            pos = counts > 0
            n_pos = int(pos.sum())
            n_neg = int((~pos).sum())
            # Give positive and zero-count scenes equal total sampling weight.
            w = np.where(pos, 0.5 / max(n_pos, 1), 0.5 / max(n_neg, 1))
            # full_n draws per epoch: subset runs keep the full-data number of steps.
            sampler = WeightedRandomSampler(w.tolist(), num_samples=full_n, replacement=True)
            print(f'[data] {s}: balanced sampler, n_pos={n_pos} n_neg={n_neg}', flush=True)
            ds = PairedFlip(ds)
        loaders[s] = DataLoader(ds,
                                batch_size=args.batch_size,
                                sampler=sampler,
                                num_workers=args.num_workers,
                                collate_fn=collate_chips,
                                pin_memory=device == 'cuda')
    return loaders


def main() -> None:
    """Build the datasets and train the selected count baseline."""
    ap, args = parse_args()
    if args.ckpt:
        args.ckpt = str(Path(args.ckpt).expanduser().resolve())
    if args.batch_size * args.accum != 32:
        print('Note: effective batch differs from the paper recipe (32).', flush=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.train_subset is not None and args.train_chips:
        ap.error('Use either --train-subset or --train-chips, not both')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[env] torch={torch.__version__} device={device} encoder={args.encoder}', flush=True)
    counts_dir = Path(args.counts_dir)
    split_index = load_split_index(args.split_csv)
    excluded = (read_unreadable_scenes(args.unreadable_scenes) if args.unreadable_scenes
                else frozenset())
    # Every image must be readable except in the listed unreadable scenes, and every listed
    # scene must have an unreadable image.
    issues = find_unreadable_images(split_index)
    unexpected = [i for i in issues if i['chip_id'] not in excluded]
    readable = sorted(excluded - {i['chip_id'] for i in issues})
    if unexpected or readable:
        raise RuntimeError('Image check failed: ' + json.dumps(
            {'unreadable_images': unexpected, 'listed_scenes_with_readable_images': readable}))
    encoder = build_encoder(args.encoder, ckpt=args.ckpt, image_size=args.image_size)
    print(f'[encoder] {args.encoder} feat_dim={encoder.feat_dim} image_size={args.image_size}',
          flush=True)
    if args.activation_checkpointing:
        if args.encoder == 'dinov3':
            encoder.vit.set_grad_checkpointing(True)
        elif args.encoder in ('remoteclip', 'skyclip'):
            encoder.clip.visual.set_grad_checkpointing(True)
        else:
            raise ValueError('Activation checkpointing is available only for dinov3, remoteclip '
                             'and skyclip; use --no-activation-checkpointing')
    model = ChipCountModel(encoder).to(device)
    run_info = {'unreadable_scenes': sorted(excluded)}
    loaders = build_loaders(args, encoder, counts_dir, split_index, excluded, run_info, device)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if 'train_subset' in run_info:
        chip_ids = run_info['train_subset']['chip_ids']
        (out / 'train_chips.txt').write_text('\n'.join(chip_ids) + '\n')
    config = {**vars(args), 'device': device, 'run_info': run_info}
    (out / 'config.json').write_text(json.dumps(config, indent=2, default=str))
    fit(model, loaders, encoder=args.encoder, question=args.question, epochs=args.epochs,
        lr=args.lr, weight_decay=args.weight_decay, device=device, out_dir=args.output,
        amp=bool(args.amp), accum=max(1, int(args.accum)), run_info=run_info)


if __name__ == '__main__':
    main()
