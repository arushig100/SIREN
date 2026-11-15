"""Train a residential or vegetation classifier on cached pre-disaster image features from base.pth.
Only this classifier's projection and head are trained. They are saved as classifier.pth; the
settings, results and image hashes go to classifier.json."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import f1_score

from siren.files import expand_data_path, require_file, sha256 as sha
from siren.settings import CLASSIFIER_THRESHOLD_GRID


def digest(t):
    """SHA-256 of a tensor's values."""
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def parse_args():
    """Read the classifier, checkpoint, label file and output path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', required=True,
                        help='Label JSON, for example data/labels/residential.json')
    parser.add_argument('--output', required=True, help='New output folder')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--base-checkpoint', type=Path, required=True,
                        help='base.pth, which supplies the frozen features')
    parser.add_argument('--concept', choices=('residential', 'vegetation'), required=True)
    parser.add_argument('--device', default=None,
                        help='torch device (default: cuda if available, else cpu)')
    args = parser.parse_args()
    args.device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    return args


def cache_features(model, rows, name, device):
    """Extract frozen pre-disaster entity features once for each scene."""
    entities = {}
    groups = defaultdict(list)
    for r in rows:
        key = (r['chip_id'], r['entity_id'])
        if key not in entities:
            entities[key] = r
            groups[r['chip_id']].append(key)
    cache = {}
    image_hashes = {}
    with torch.inference_mode():
        for (num, (chip, keys)) in enumerate(sorted(groups.items()), 1):
            rr = [entities[k] for k in keys]
            boxes = torch.tensor([r['bbox'] for r in rr], device=device, dtype=torch.float32)
            for side in ('pre_image_path', 'post_image_path'):
                if any(r[side] != rr[0][side] for r in rr):
                    raise ValueError(f'Rows of scene {chip} list different {side} values')
                # Hash both images; the classifiers read only pre-disaster features.
                image_hashes[rr[0][side]] = sha(require_file(rr[0][side]))
            with Image.open(rr[0]['pre_image_path']) as im:
                if im.size != (1024, 1024):
                    raise ValueError(f"Expected a 1024x1024 image: {rr[0]['pre_image_path']}")
                a = np.asarray(im.convert('RGB')).copy()
            tensor = torch.from_numpy(a).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
            # Pooled 7x7 features with the classifier's context padding: (N, 6272).
            features = model.pooled(model.fpn(tensor), boxes, name).cpu().numpy()
            for (i, k) in enumerate(keys):
                cache[k] = features[i]
            print(f'Cached {num}/{len(groups)} scenes', flush=True)
    return (cache, image_hashes)


def train_classifier(model, rows, name, cache, device):
    """Fit the projection and classifier, then select a validation threshold.

    Projection 6272 -> 64 and head 64 -> 32 -> 2, both reset; 30 epochs of AdamW (1e-3),
    batch 256."""
    proj = model.concept_proj[name]
    head = model.heads[name]
    for module in (proj, head):
        for layer in module.modules():
            if isinstance(layer, torch.nn.Linear):
                layer.reset_parameters()
        for p in module.parameters():
            p.requires_grad_(True)
    rr = [r for r in rows if r['concept'] == name]
    tr = [r for r in rr if r['split'] == 'train']
    va = [r for r in rr if r['split'] == 'val']
    assert not {r['chip_id'] for r in tr} & {r['chip_id'] for r in va}

    def tensors(rs):
        """Load the selected cached features and labels onto the training device."""
        return (torch.tensor(np.stack([cache[r['chip_id'], r['entity_id']] for r in rs]),
                             device=device),
                torch.tensor([int(r['label'] == 'yes') for r in rs], device=device))

    (pre, y) = tensors(tr)
    (vp, vy) = tensors(va)
    assert len(torch.unique(y)) == len(torch.unique(vy)) == 2

    def logits(p):
        """Project entity features and apply the selected classifier."""
        return head(proj(p))

    # Inverse-square-root class weights, normalized to mean 1.
    weights = torch.bincount(y, minlength=2).float().rsqrt()
    weights /= weights.mean()
    opt = torch.optim.AdamW(list(proj.parameters()) + list(head.parameters()),
                            lr=0.001,
                            weight_decay=0.0001)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
    best = -1
    history = []
    for epoch in range(30):
        proj.train()
        head.train()
        order = torch.randperm(len(y), device=device)
        for start in range(0, len(y), 256):
            idx = order[start:start + 256]
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(logits(pre[idx]),
                                                     y[idx],
                                                     weight=weights)
            loss.backward()
            opt.step()
        sched.step()
        proj.eval()
        head.eval()
        with torch.no_grad():
            pred = logits(vp).argmax(1).cpu().numpy()
        score = float(f1_score(vy.cpu().numpy(), pred, average='macro'))
        history.append(score)
        if score > best:  # keep the epoch with the best validation macro-F1
            best = score
            best_state = ({
                k: v.detach().cpu().clone()
                for (k, v) in proj.state_dict().items()
            }, {
                k: v.detach().cpu().clone()
                for (k, v) in head.state_dict().items()
            })
    proj.load_state_dict(best_state[0])
    head.load_state_dict(best_state[1])
    with torch.no_grad():
        prob = logits(vp).softmax(1)[:, 1].cpu().numpy()
    # Calibrate on validation labels after choosing the best epoch.
    grid = CLASSIFIER_THRESHOLD_GRID
    f1s = [
        float(f1_score(vy.cpu().numpy(), (prob >= t).astype(int), average='macro'))
        for t in grid
    ]
    # Plain Python numbers for the JSON report.
    thr = float(grid[int(np.argmax(f1s))])
    result = dict(
        n_train=len(tr),
        n_val=len(va),
        best_val_macro_f1=best,
        epoch_scores=history,
        validation_threshold=thr,
        validation_threshold_macro_f1=max(f1s),
        threshold_rule='argmax per-entity macro-F1 on validation',
        threshold_grid=[float(t) for t in grid])
    print(name, result, flush=True)
    return result


def main():
    """Cache entity features, train the classifier and save its best checkpoint."""
    args = parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(f'{out} already exists; choose a new output folder')
    data = json.loads(Path(args.labels).read_text())
    rows = data['rows']
    residential = args.concept == 'residential'
    expected_schema = 'residential_labels' if residential else 'vegetation_labels'
    name = 'is_residential' if residential else 'surrounded_by_vegetation'
    if data.get('schema') != expected_schema:
        raise ValueError(f"--concept {args.concept} needs labels with schema {expected_schema}, "
                         f"got {data.get('schema')}")
    if any(r['split'] not in ('train', 'val') for r in rows):
        raise ValueError('Every label row must be in the train or val split')
    if {r['concept'] for r in rows} != {name}:
        raise ValueError(f'Every label row must have concept {name}')
    # Image paths may use ${DM3_ROOT} or be relative to the label file.
    for row in rows:
        for field in ('pre_image_path', 'post_image_path'):
            path = Path(expand_data_path(row[field])).expanduser()
            row[field] = str(path if path.is_absolute() else Path(args.labels).resolve().parent /
                             path)
    from siren.concepts.registry import get_concepts
    from siren.perception.model import SharedPerceptionModel
    base = args.base_checkpoint
    # Fixed seed and deterministic kernels.
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # base.pth supplies every weight except the new classifier's, so the pretrained Satlas
    # weights are not downloaded.
    model = SharedPerceptionModel(concepts=get_concepts(['is_road_blocked', name]),
                                  pretrained_backbone=False)
    prefixes = (f'concept_proj.{name}.', f'heads.{name}.')
    state = torch.load(base, map_location='cpu', weights_only=True)['model_state_dict']
    # The new classifier keeps its fresh weights; every other weight comes from base.pth.
    state.update({k: v for (k, v) in model.state_dict().items() if k.startswith(prefixes)})
    model.load_state_dict(state, strict=True)
    del state
    # Training must leave every other weight unchanged.
    before = {k: digest(v) for (k, v) in model.state_dict().items() if not k.startswith(prefixes)}
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(args.device).eval()
    (cache, image_hashes) = cache_features(model, rows, name, args.device)
    result = train_classifier(model, rows, name, cache, args.device)
    if any(digest(v) != before[k] for (k, v) in model.state_dict().items() if k in before):
        raise RuntimeError('Training changed weights outside the classifier')
    config = dict(seed=args.seed,
                  epochs=30,
                  lr=0.001,
                  weight_decay=0.0001,
                  batch_size=256,
                  balance='inverse_sqrt',
                  labels_sha256=sha(args.labels),
                  base_checkpoint_sha256=sha(base))
    out.mkdir(parents=True)
    # Only the trained classifier's tensors; inference loads them on top of base.pth.
    torch.save(
        dict(model_state_dict={k: v.cpu()
                               for (k, v) in model.state_dict().items() if k.startswith(prefixes)}),
        out / 'classifier.pth')
    (out / 'classifier.json').write_text(
        json.dumps(dict(config=config, results={name: result}, image_hashes=image_hashes),
                   indent=2))


if __name__ == '__main__':
    main()
