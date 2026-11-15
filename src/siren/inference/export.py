"""Export scored buildings and roads from paired images: one entity JSON per scene.

The export checks the model files and images, locks the output folder, records the run
settings and resumes an interrupted run with the same settings."""
import argparse
from contextlib import contextmanager
import csv
import fcntl
import json
from pathlib import Path
import tempfile

from siren.files import expand_data_path, read_unreadable_scenes, sha256 as sha
from siren.settings import ROAD_MASK_COLUMNS
from .pipeline import ASSET_KEYS, check_hashes, load_assets, pipeline_from_assets

# Source files of the package; their hashes are part of the run settings.
SOURCE_ROOT = Path(__file__).resolve().parents[1]
RUN_SETTINGS = '_run_settings.json'


def atomic_json(path, value):
    """Write JSON through a temporary file and a rename, so no reader sees a partial file."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, suffix='.tmp',
                                     delete=False) as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        temporary = Path(stream.name)
    temporary.replace(path)


def select_rows(split_csv, split, data_root, excluded, scene_ids=None):
    """Return the split's scenes, sorted by ID, without excluded scenes and with absolute paths.

    Empty road-mask paths stay empty."""
    with Path(split_csv).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    ids = [r['chip_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate chip IDs in the split file')
    selected = []
    for row in rows:
        if row['split'] != split or row['chip_id'] in excluded:
            continue
        cid = row['chip_id']
        # Scene IDs become output file names, so they may not contain path separators.
        if not cid or Path(cid).name != cid or cid in ('.', '..') or '\\' in cid:
            raise ValueError(f'Unsafe chip ID: {cid!r}')
        if scene_ids is not None and cid not in scene_ids:
            continue
        # ${DM3_ROOT} in the split file stands for --data-root, or the DM3_ROOT variable if unset.
        for key in ('pre_image_path', 'post_image_path', *ROAD_MASK_COLUMNS):
            if key not in row:
                continue
            value = expand_data_path(row[key], data_root)
            if (key.endswith('image_path') and not value) or (value and (
                    '${' in value or not Path(value).is_absolute())):
                raise ValueError(f'Unresolved path: {value!r}')
            row[key] = value
        selected.append(row)
    selected.sort(key=lambda r: r['chip_id'])
    if scene_ids is not None and {r['chip_id'] for r in selected} != set(scene_ids):
        raise ValueError('Requested IDs missing, excluded or outside selected split')
    if not selected:
        raise ValueError('Empty scene selection')
    return selected


def check_images(rows, check_masks=False):
    """Open every image (and, with check_masks, every listed road mask); return image hashes."""
    from PIL import Image
    images = {}
    for row in rows:
        record = []
        for key in ('pre_image_path', 'post_image_path'):
            path = row[key]
            if not Path(path).is_file():
                raise FileNotFoundError(f'Image not found: {path}. Check --data-root: it must be '
                                        'the DisasterM3 folder with DisasterM3_Bench/ and '
                                        'DisasterM3_Instruct/')
            with Image.open(path) as im:
                if im.size != (1024, 1024):
                    raise ValueError(f'Expected 1024x1024 image: {path}')
                im.convert('RGB').load()
            record.append({'path': path, 'sha256': sha(path)})
        images[row['chip_id']] = record
        if check_masks:
            for key in ROAD_MASK_COLUMNS:
                if row.get(key):
                    with Image.open(row[key]) as im:
                        im.load()
    return images


def prepare(args):
    """Check assets and input images, then return them with the run settings.

    The run settings record everything that determines the outputs; a resumed run must match
    them. --preflight also opens the road masks listed in the split file."""
    assets = load_assets(args.assets)
    check_hashes(assets, ASSET_KEYS)
    excluded = read_unreadable_scenes(assets['unreadable_scenes']['path'])
    requested = set(args.scene_ids) if args.scene_ids else None
    if args.scene_ids and len(requested) != len(args.scene_ids):
        raise ValueError('Duplicate requested scene IDs')
    rows = select_rows(assets['split_csv']['path'], args.split, args.data_root.resolve(), excluded,
                       requested)
    images = check_images(rows, check_masks=args.preflight)
    sources = sorted(SOURCE_ROOT.rglob('*.py'))
    # Record the library versions; geometry and model outputs depend on them.
    import cv2
    import numpy
    import scipy
    import shapely
    import skimage
    import torch
    import torchvision
    settings = {
        'schema': 'siren_export',
        'assets': assets,
        'split': args.split,
        'scene_ids': [r['chip_id'] for r in rows],
        'rows': rows,
        'images': images,
        'source_sha256': {str(p): sha(p) for p in sources},
        'runtime': {
            'torch': str(torch.__version__),
            'torchvision': str(torchvision.__version__),
            'numpy': str(numpy.__version__),
            'scipy': str(scipy.__version__),
            'scikit-image': str(skimage.__version__),
            'opencv': str(cv2.__version__),
            'shapely': str(shapely.__version__),
            'device': args.device
        },
    }
    return assets, rows, settings


@contextmanager
def locked_output(output, settings):
    """Hold an exclusive lock on the output folder and check its saved run settings."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / '_lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another process owns this output folder') from error
        saved = output / RUN_SETTINGS
        if saved.exists():
            if json.loads(saved.read_text()) != settings:
                raise ValueError('Output folder was written with different settings; '
                                 'use a new folder')
        elif any(p.name != '_lock' for p in output.iterdir()):
            raise ValueError('Output folder has files from another export; use a new folder')
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def validate_outputs(output, settings):
    """Check that every scene file is tracked in _progress.json with its hash; return chip ID -> hash."""
    progress_path = output / '_progress.json'
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    wanted = set(settings['scene_ids'])
    found = {p.stem for p in output.glob('*.json') if not p.name.startswith('_')}
    if found != set(progress) or not found <= wanted:
        raise ValueError('Untracked, missing or unexpected scene outputs')
    for cid, digest in progress.items():
        path = output / f'{cid}.json'
        if sha(path) != digest:
            raise ValueError(f'Output hash mismatch: {cid}')
        data = json.loads(path.read_text())
        if data['chip_id'] != cid or data['split'] != settings['split']:
            raise ValueError(f'Output scene identity mismatch: {cid}')
    complete = output / '_complete.json'
    if complete.exists() and (found != wanted or json.loads(complete.read_text()) != progress):
        raise ValueError('Invalid completion record')
    return progress


def export(args):
    """Write one entity JSON per selected scene; resume only a run with the same settings."""
    if args.output is None and not args.preflight:
        raise ValueError('--output is required unless --preflight is used')
    assets, rows, settings = prepare(args)
    if args.preflight:
        print(json.dumps({'validated_scenes': len(rows), 'split': settings.get('split'),
                          'runtime': settings.get('runtime')}, indent=2))
        return
    with locked_output(args.output, settings):
        progress = validate_outputs(args.output, settings)
        remaining = [r for r in rows if r['chip_id'] not in progress]
        if remaining:
            import numpy as np
            from PIL import Image
            import torch
            # CPU results can depend on the thread count; the saved predictions used two threads.
            torch.set_num_threads(2)
            pipeline = pipeline_from_assets(assets, args.device)
            # Saved only after the model loads, so a failed start leaves the folder reusable.
            if not (args.output / RUN_SETTINGS).exists():
                atomic_json(args.output / RUN_SETTINGS, settings)
            for row in remaining:
                images = []
                # RGB images as float32 tensors of shape (3, 1024, 1024) in [0, 1].
                for key in ('pre_image_path', 'post_image_path'):
                    with Image.open(row[key]) as im:
                        images.append(
                            torch.from_numpy(np.array(im.convert('RGB'))).permute(
                                2, 0, 1).float().div(255).to(args.device))
                cid = row['chip_id']
                result = pipeline.run(*images, chip_id=cid, split=args.split,
                                      disaster_id=row.get('disaster_id', ''))
                path = args.output / f'{cid}.json'
                atomic_json(path, result)
                progress[cid] = sha(path)
                atomic_json(args.output / '_progress.json', progress)
                print(f'{len(progress)}/{len(rows)} {cid}', flush=True)
        validate_outputs(args.output, settings)
        atomic_json(args.output / '_complete.json', progress)


def default_device():
    """Return 'cuda' when a GPU is available, else 'cpu'."""
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    """Check the inputs, load the model and export the selected image pairs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True,
                        help='asset config, for example configs/assets.json')
    parser.add_argument('--data-root', type=Path, required=True,
                        help='DisasterM3 folder that replaces ${DM3_ROOT} in the split file')
    parser.add_argument('--split', choices=['train', 'val', 'test'], required=True,
                        help='split to export')
    parser.add_argument('--scene-ids', nargs='+', help='export only these scene IDs')
    parser.add_argument('--device', default=None,
                        help='torch device (default: cuda if available, else cpu)')
    parser.add_argument('--output', type=Path,
                        help='output folder; an interrupted export resumes here')
    parser.add_argument('--preflight', action='store_true',
                        help='check the assets, images and road masks without running the model')
    args = parser.parse_args()
    args.device = args.device or default_device()
    export(args)


if __name__ == '__main__':
    main()
