"""Measure inference time, FLOPs and used parameters on supplied image pairs."""
import argparse
import json
import time
from pathlib import Path

from siren.evaluation.external import count_scene, load_baseline, pairs
from siren.files import sha256
from siren.programs.recipes import COUNT_QUERIES


def parse_args():
    """Read image-pair paths and the models to measure."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pairs',
                    type=Path,
                    required=True,
                    help='Image-pair CSV (chip_id,pre_image_path,post_image_path); the paper '
                    'timed the 300 test scenes listed in data/subsets/timing_300.txt')
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument('--siren-assets', type=Path,
                        help='SIREN asset config, such as configs/assets.json')
    source.add_argument('--baseline-runs',
                        type=Path,
                        nargs=8,
                        help='Eight run directories, one for each paper query')
    ap.add_argument('--output', type=Path, required=True,
                    help='Output JSON file; must not exist yet')
    ap.add_argument('--device', default=None,
                    help='torch device (default: cuda if available, else cpu)')
    ap.add_argument('--profile-scenes',
                    type=int,
                    help='Count FLOPs and used parameters on the first N pairs only '
                    '(default: all pairs, as in the paper)')
    args = ap.parse_args()
    if args.device is None:
        import torch
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return args


def baseline_questions(runs):
    """Read the question of each baseline run before any model is loaded or timed."""
    questions = [json.loads((Path(run) / 'config.json').read_text())['question']
                 for run in runs]
    if set(questions) != set(COUNT_QUERIES):
        raise ValueError(f'Baseline runs must cover each paper query once; got {questions}')
    return questions


def measure(predict, modules, load, rows, sync, profile_scenes=None):
    """Warm up the model, time inference, then count FLOPs and used parameters.

    load(i) returns pair i on the device. It runs outside the timed region, and pairs are
    loaded one at a time rather than all kept on the device.
    """
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.utils.flop_counter import FlopCounterMode
    with torch.inference_mode():
        # Three warm-up pairs, then one timed pass over every pair.
        for i in range(min(3, len(rows))):
            predict(load(i), rows[i])
        sync()
        seconds = []
        for (i, row) in enumerate(rows):
            pair = load(i)
            sync()
            start = time.perf_counter()
            predict(pair, row)
            sync()
            seconds.append(time.perf_counter() - start)
        # A parameter is used if its module ran or if it was passed to a tensor operation;
        # the second case catches functional calls such as F.linear(x, self.qkv.weight).
        parameters = {id(p): p for root in modules for p in root.parameters()}
        pointers = {p.data_ptr() for p in parameters.values()}
        (ran, passed) = (set(), set())

        class ParameterUse(TorchDispatchMode):

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                """Record parameters passed directly to tensor operations."""
                kwargs = kwargs or {}
                for value in (*args, *kwargs.values()):
                    for t in (value if isinstance(value, (list, tuple)) else (value, )):
                        if isinstance(t, torch.Tensor) and t.data_ptr() in pointers:
                            passed.add(t.data_ptr())
                return func(*args, **kwargs)

        modules_by_id = {id(m): m for root in modules for m in root.modules()}
        hooks = [
            m.register_forward_pre_hook(lambda module, inputs: ran.add(module))
            for m in modules_by_id.values()
        ]
        fastpath = torch.backends.mha.get_fastpath_enabled()
        # Fused attention operations (the MultiheadAttention fast path and, on CPU, the fused
        # scaled dot-product kernel) are not counted, so use the plain math version while counting.
        torch.backends.mha.set_fastpath_enabled(False)
        flops = []
        try:
            for i in range(len(rows) if profile_scenes is None else profile_scenes):
                pair = load(i)
                with sdpa_kernel(SDPBackend.MATH), FlopCounterMode(display=False) as counter, ParameterUse():
                    predict(pair, rows[i])
                flops.append(counter.get_total_flops())
        finally:
            torch.backends.mha.set_fastpath_enabled(fastpath)
            for hook in hooks:
                hook.remove()
    owned = {id(p) for m in ran for p in m.parameters(recurse=False)}
    used = {
        key: p.numel()
        for (key, p) in parameters.items() if key in owned or p.data_ptr() in passed
    }
    return (seconds, flops, used)


def main():
    """Measure the computation required to answer all benchmark queries."""
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f'{args.output} already exists; choose a new output file')
    questions = baseline_questions(args.baseline_runs) if args.baseline_runs else [None]
    import numpy as np
    import torch
    from PIL import Image
    rows = pairs(args.pairs)
    if args.profile_scenes is not None and not 1 <= args.profile_scenes <= len(rows):
        raise ValueError('--profile-scenes must be between 1 and the number of pairs')
    # Load images before timing; preprocessing is also excluded below.
    images = []
    for row in rows:
        pair = []
        for key in ('pre_image_path', 'post_image_path'):
            with Image.open(row[key]) as im:
                pair.append(im.convert('RGB').copy())
        images.append(pair)

    def sync():
        """Wait for queued GPU operations before reading the timer."""
        if args.device.startswith('cuda'):
            torch.cuda.synchronize(args.device)

    measurements = []
    for (run, question) in zip(args.baseline_runs or [None], questions):
        if run is None:
            from siren.inference.pipeline import load_pipeline
            model = load_pipeline(args.siren_assets, args.device)
            modules = [model.model, *model.building.heads.values()]
            transform = lambda im: torch.from_numpy(np.array(im)).permute(2, 0, 1).float().div(255)

            def predict(pair, row):
                """Answer all eight count queries for one image pair."""
                record = model.run(*pair, chip_id=row['chip_id'], split='test')
                return [count_scene(record, q) for q in COUNT_QUERIES]

            name = 'SIREN'
        else:
            (model, transform, _) = load_baseline(run, args.device)
            modules = [model]

            def predict(pair, row):
                """Run this baseline model on one image pair."""
                return model(*(im.unsqueeze(0) for im in pair))

            name = str(run)
        # Preprocess on the CPU before timing; each pair moves to the device when it is used.
        tensors = [[transform(im) for im in pair] for pair in images]

        def load(i):
            """Move one preprocessed image pair onto the selected device."""
            return [t.to(args.device) for t in tensors[i]]

        (seconds, flops, used) = measure(predict, modules, load, rows, sync, args.profile_scenes)
        measurements.append({
            'model': name,
            'question': question,
            'seconds_per_scene': seconds,
            'flops_per_profiled_scene': flops,
            'used_parameters': sum(used.values())
        })
        # Drop every reference to this model so its memory is freed before the next one loads.
        del predict, load
        (model, modules, tensors) = (None, None, None)
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()
    # Totals for all eight questions: SIREN is one model, a baseline is the sum of eight models.
    result = {
        'n_scenes': len(rows),
        'n_queries': len(COUNT_QUERIES),
        'seconds_mean': float(sum((np.mean(m['seconds_per_scene']) for m in measurements))),
        'flops_mean': float(sum((np.mean(m['flops_per_profiled_scene']) for m in measurements))),
        'parameters': sum((m['used_parameters'] for m in measurements)),
        'device': torch.cuda.get_device_name(args.device)
        if args.device.startswith('cuda') else args.device,
        'torch': torch.__version__,
        'manifest_sha256': sha256(args.pairs),
        'measurements': measurements,
        'notes': ('FLOPs: FlopCounterMode operations, two per multiply-accumulate, attention '
                  'counted with the math kernel. Used parameters: modules that ran plus tensors passed to operations. '
                  'Times: all eight queries, without image loading, preprocessing or device '
                  'transfer; programs are fixed, so no language model runs.')
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
