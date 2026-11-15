"""Run SIREN or a trained count baseline on external-event image pairs.
Save scene count predictions; score them with siren.evaluation.counts."""
import argparse
import csv
import json
from pathlib import Path

from siren.files import expand_data_path, sha256
from siren.metrics import round_count
from siren.programs.recipes import COUNT_QUERIES
from siren.settings import IMAGE_SIZE


def pairs(path):
    """Read an image-pair CSV; paths may use environment variables or be relative to the CSV."""
    with Path(path).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    ids = [r['chip_id'] for r in rows]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Manifest must contain distinct, nonempty scene IDs')
    for r in rows:
        chip = r['chip_id']
        if not chip or Path(chip).name != chip or chip in ('.', '..'):
            raise ValueError('Invalid scene ID')
        for key in ('pre_image_path', 'post_image_path'):
            p = Path(expand_data_path(r[key])).expanduser()
            if not p.is_absolute():
                p = Path(path).resolve().parent / p
            if not p.is_file():
                raise FileNotFoundError(p)
            r[key] = str(p)
    return rows


def load_baseline(run, device):
    """Load a trained count baseline, its evaluation transform and its saved config."""
    import torch
    from siren.baselines.encoders import build_encoder
    from siren.baselines.model import ChipCountModel
    run = Path(run)
    cfg = json.loads((run / 'config.json').read_text())
    ckpt = cfg.get('ckpt')
    if ckpt and (not Path(ckpt).is_absolute()):
        ckpt = str((run / ckpt).resolve())
    encoder = build_encoder(cfg['encoder'], ckpt=ckpt, image_size=cfg['image_size'])
    model = ChipCountModel(encoder).to(device)
    saved = torch.load(run / 'best.pth', map_location=device, weights_only=True)
    model.load_state_dict(saved['model'], strict=True)
    model.eval()
    return (model, encoder.transform(cfg['image_size']), cfg)


def count_scene(scene, question):
    """Run the question's counting program on one scene's entity record."""
    # The 50 px "near" and 2,000 px² "large" rules are in DisasterM3 pixels (0.8 m per
    # pixel), so at another image scale they mean other distances and areas.
    # blocked_roads_near_homes instead takes a distance in metres and a pixel size.
    from siren.programs.engine import execute
    from siren.programs.recipes import count_program
    return execute(count_program(question), scene)['value']


def parse_args():
    """Read the image manifest, selected model and output path."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pairs', type=Path, required=True)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument('--siren-assets', type=Path)
    source.add_argument('--baseline-run', type=Path)
    ap.add_argument('--question',
                    choices=COUNT_QUERIES,
                    required=True,
                    help='Count query; distances and areas are in pixels (50 px, 2,000 px², set '
                    'for DisasterM3)')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--device', default=None,
                    help='torch device (default: cuda if available, else cpu)')
    args = ap.parse_args()
    if args.device is None:
        import torch
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return args


def load_predictor(args):
    """Load SIREN or the count model trained for the requested question."""
    import numpy as np
    import torch
    if args.siren_assets:
        from siren.inference.pipeline import load_pipeline
        model = load_pipeline(args.siren_assets, args.device)
        transform = lambda im: torch.from_numpy(np.array(im)).permute(2, 0, 1).float().div(255)
        system = 'SIREN'
    else:
        (model, transform, cfg) = load_baseline(args.baseline_run, args.device)
        if cfg['question'] != args.question:
            raise ValueError('The checkpoint was trained for a different question')
        system = cfg['encoder']
    return (model, transform, system)


def predict_scenes(args, rows):
    """Predict each scene count and save SIREN's entity records."""
    import numpy as np
    import torch
    from PIL import Image
    (model, transform, system) = load_predictor(args)
    args.output.mkdir(parents=True)
    predictions = []
    with torch.inference_mode():
        for row in rows:
            images = []
            for key in ('pre_image_path', 'post_image_path'):
                with Image.open(row[key]) as im:
                    if args.siren_assets and im.size != (IMAGE_SIZE, IMAGE_SIZE):
                        raise ValueError('SIREN needs 1024 x 1024-pixel images')
                    images.append(transform(im.convert('RGB')).to(args.device))
            if args.siren_assets:
                scene = model.run(*images,
                                  chip_id=row['chip_id'],
                                  split='external',
                                  disaster_id=row.get('disaster_id', ''))
                (args.output / (row['chip_id'] + '.json')).write_text(json.dumps(scene))
                raw = float(count_scene(scene, args.question))
            else:
                raw = float(model(*(im.unsqueeze(0) for im in images)).item())
            if not np.isfinite(raw):
                raise ValueError('Nonfinite prediction')
            predictions.append(dict(chip_id=row['chip_id'], prediction=round_count(raw), raw=raw))
    return (predictions, system)


def write_results(args, rows, predictions, system):
    """Write predictions.csv and run_info.json (input, image and model hashes)."""
    with (args.output / 'predictions.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['chip_id', 'prediction', 'raw'])
        writer.writeheader()
        writer.writerows(predictions)
    run_info = {
        'question': args.question,
        'system': system,
        'manifest_sha256': sha256(args.pairs),
        'images': {
            r['chip_id']: {
                k: sha256(r[k])
                for k in ('pre_image_path', 'post_image_path')
            }
            for r in rows
        }
    }
    if args.baseline_run:
        run_info['checkpoint_sha256'] = sha256(args.baseline_run / 'best.pth')
    else:
        run_info['assets_sha256'] = sha256(args.siren_assets)
    (args.output / 'run_info.json').write_text(json.dumps(run_info, indent=2))


def main():
    """Run the selected model and save its scene-level count predictions."""
    args = parse_args()
    rows = pairs(args.pairs)
    if args.output.exists():
        raise FileExistsError(f'{args.output} already exists; choose a new output folder')
    (predictions, system) = predict_scenes(args, rows)
    write_results(args, rows, predictions, system)


if __name__ == '__main__':
    main()
