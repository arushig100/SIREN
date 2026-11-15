"""Train the shared encoder, building detector and road segmentation head (Stage 1), then the
blocked-road classifier on entity features from the fixed encoder (Stage 2)."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from siren.files import read_unreadable_scenes
from siren.metrics import f1_score
from siren.perception.data import (MultiTaskChipDataset, collate_multitask, load_concept_supervision,
                                   load_image_chw)
from siren.perception.eval import building_f1, road_f1
from siren.perception.model import POOLED_DIM, SharedPerceptionModel, _combine
from siren.segmentation.building.model import maskrcnn_targets
from siren.segmentation.road.losses import roads_loss
from siren.settings import IMAGE_SIZE


def make_lr_lambda(total_opt_steps, warmup_opt_steps):
    """Warmup followed by cosine decay, indexed by optimizer steps."""

    def fn(opt_step):
        """Return the learning-rate multiplier for this optimizer step."""
        if opt_step < warmup_opt_steps:
            return float(opt_step) / float(max(1, warmup_opt_steps))
        progress = (opt_step - warmup_opt_steps) / max(1, total_opt_steps - warmup_opt_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return fn


def class_balanced_weight(counts: Counter, n_classes: int = 2, device: str = "cpu") -> torch.Tensor:
    """Inverse-square-root class weights: sqrt(total / (n_classes * count))."""
    total = sum(counts.values())
    w = [math.sqrt(total / (n_classes * max(1, counts.get(c, 0)))) for c in range(n_classes)]
    return torch.tensor(w, dtype=torch.float32, device=device)


def f1_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Binary accuracy, per-class F1 and their mean (macro F1)."""

    def _f1_for_class(c):
        """Calculate F1 with the selected class treated as positive."""
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        return f1_score(tp, fp, fn)

    f1_pos = _f1_for_class(1)
    f1_neg = _f1_for_class(0)
    return {
        "n": int(len(y_true)),
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else 0.0,
        "f1_pos": f1_pos,
        "f1_neg": f1_neg,
        "macro_f1": (f1_pos + f1_neg) / 2,
    }


def load_warmstart(model: SharedPerceptionModel, detector_checkpoint: str):
    """Start the detector, encoder included, from a building-detector checkpoint."""
    sd = torch.load(detector_checkpoint, map_location="cpu", weights_only=True)["model_state_dict"]
    missing, unexpected = model.det.load_state_dict(sd, strict=False)
    n_bb = sum(1 for k in missing if "satlas_model" in k)
    print(
        f"[warmstart] loaded building detector into model.det "
        f"(missing={len(missing)} [encoder-missing={n_bb}], unexpected={len(unexpected)})",
        flush=True)
    if n_bb > 0:
        raise RuntimeError(f"[warmstart] {detector_checkpoint} is missing {n_bb} encoder weights; "
                           "Stage 1 must start from the building detector.")


def _split_lr_groups(model, backbone_lr: float, head_lr: float):
    """Use separate learning rates for pretrained encoder and task parameters."""
    bb, head = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_trunk = "satlas_model" in n
        (bb if is_trunk else head).append(p)
    return [{"params": bb, "lr": backbone_lr}, {"params": head, "lr": head_lr}]


def stage1(args, model, device, run_dir: Path):
    """Train the encoder, building detector and road segmentation head together on whole scenes."""
    train_ds = MultiTaskChipDataset(split_csv=args.split_csv,
                                    building_entities_dir=args.building_entities_dir,
                                    split="train",
                                    image_size=IMAGE_SIZE)
    val_ds = MultiTaskChipDataset(split_csv=args.split_csv,
                                  building_entities_dir=args.building_entities_dir,
                                  split="val",
                                  image_size=IMAGE_SIZE,
                                  unreadable_scenes=args.unreadable_scenes)
    if args.max_val_samples:
        val_ds.sample_scenes(args.max_val_samples, args.seed)
    train_loader = DataLoader(train_ds,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              collate_fn=collate_multitask,
                              drop_last=True,
                              pin_memory=True)
    val_loader = DataLoader(val_ds,
                            batch_size=max(1, args.batch_size),
                            shuffle=False,
                            num_workers=max(1, args.num_workers // 2),
                            collate_fn=collate_multitask)
    print(f"[stage1] train={len(train_ds)} val={len(val_ds)}", flush=True)

    opt = torch.optim.AdamW(_split_lr_groups(model, args.backbone_lr, args.head_lr),
                            weight_decay=args.weight_decay)
    # The schedule counts optimizer updates: 40,000 minibatches / 4 accumulation steps = 10,000.
    total_opt = args.num_iter // args.grad_accum
    warmup_opt = args.warmup_iter // args.grad_accum
    sched = torch.optim.lr_scheduler.LambdaLR(opt, make_lr_lambda(total_opt, warmup_opt))
    amp_enabled = args.amp == "bf16"
    amp_dtype = torch.bfloat16

    step = 0
    data_iter = iter(train_loader)
    model.train()
    while step < args.num_iter:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        pre = batch["pre"].to(device, non_blocking=True)

        # Detection loss uses scenes with reference boxes.
        targets_all = maskrcnn_targets(batch["targets"], str(device))
        keep = [i for i, t in enumerate(targets_all) if t["boxes"].numel() > 0]
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
            det_loss = pre.new_zeros(())
            if keep:
                det_in = pre[keep]
                det_tg = [targets_all[i] for i in keep]
                loss_dict = model.det(det_in, det_tg)
                det_loss = sum(v.float() for v in loss_dict.values())
            # Road loss uses scenes with road labels.
            road_loss = pre.new_zeros(())
            vmask = batch["road_valid"]
            if vmask.any():
                road_logits = model.road_logits(pre)  # (B,1,H,W)
                gt = batch["road_semantic"].to(device)  # (B,H,W)
                vidx = vmask.nonzero(as_tuple=True)[0].to(device)
                rl = roads_loss(road_logits[vidx], gt[vidx])
                road_loss = rl.float()
            # Weight the road loss by --w-road; dividing by grad_accum averages the accumulated gradients.
            loss = (det_loss + args.w_road * road_loss) / args.grad_accum

        # Skip batches with neither building nor road supervision.
        if loss.requires_grad:
            loss.backward()
        if (step + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()

        if (step + 1) % args.log_every == 0:
            print(
                f"[s1 {step+1}/{args.num_iter}] det={float(det_loss):.3f} "
                f"road={float(road_loss):.3f} lr={sched.get_last_lr()[0]:.2e}",
                flush=True)

        if (step + 1) % args.eval_every == 0 or (step + 1) == args.num_iter:
            # Validation F1 is only logged; Stage 2 starts from the last weights.
            bmet = building_f1(model, val_loader, device)
            rmet = road_f1(model, val_loader, device)
            model.train()
            print(f"[s1 eval {step+1}] bldg_F1={bmet['f1']:.4f} road_F1={rmet['f1']:.4f}",
                  flush=True)
            _save_stage1(run_dir, model, step + 1, bmet, rmet)
        step += 1
    print(f"[stage1] done -> {run_dir / 'stage1_last.pth'}", flush=True)


def _save_stage1(run_dir, model, step, bmet, rmet):
    """Save stage1_last.pth: the current weights and their validation F1."""
    ck = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "config": {"image_size": IMAGE_SIZE},
        "building_f1": bmet["f1"],
        "road_f1": rmet["f1"]
    }
    torch.save(ck, run_dir / "stage1_last.pth")


# Stage 2: cache pooled features, then train the blocked-road classifier's projection and head on them.


@torch.no_grad()
def _precompute_pooled(model, rows: list, concept, device, image_size: int):
    """Cache features and labels, retaining the matching entity order."""
    model.eval()
    by_chip: dict[str, list] = {}
    for r in rows:
        by_chip.setdefault(r.chip_id, []).append(r)
    fpre, fpost, labels, ordered_rows = [], [], [], []
    for chip_id, crows in by_chip.items():
        boxes = torch.tensor([r.bbox for r in crows], dtype=torch.float32, device=device)
        try:
            pre_t = load_image_chw(crows[0].pre_image_path, image_size, device)
        except Exception as e:
            raise ValueError(f"Cannot read pre image for {chip_id}: {crows[0].pre_image_path}") from e
        fpn_pre = model.fpn(pre_t)
        p_pre = model.pooled(fpn_pre, boxes, concept.name)  # (n,6272)
        fpre.append(p_pre.half().cpu().numpy())
        if concept.is_paired:
            # Paired classifiers also need post-disaster features.
            post_path = crows[0].post_image_path
            if not post_path:
                raise ValueError(f"Missing post image path for {chip_id}")
            try:
                post_t = load_image_chw(post_path, image_size, device)
            except Exception as e:
                raise ValueError(f"Cannot read post image for {chip_id}: {post_path}") from e
            fpn_post = model.fpn(post_t)
            p_post = model.pooled(fpn_post, boxes, concept.name)
            fpost.append(p_post.half().cpu().numpy())
        labels.extend(r.label for r in crows)
        ordered_rows.extend(crows)
    Fpre = np.concatenate(fpre, axis=0) if fpre else np.zeros((0, POOLED_DIM), np.float16)
    Fpost = np.concatenate(fpost, axis=0) if concept.is_paired and fpost else None
    y = np.asarray(labels, dtype=np.int64)
    return Fpre, Fpost, y, ordered_rows


def _train_concept(model, concept, Fpre_tr, Fpost_tr, ytr, Fpre_va, Fpost_va, yva, device,
                   args) -> tuple[dict, np.ndarray]:
    """Train one classifier's projection and head on cached pooled features."""
    name = concept.name
    proj = model.concept_proj[name].to(device)
    head = model.heads[name].to(device)
    params = list(proj.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=args.head_lr_s2, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs_s2)
    crit = nn.CrossEntropyLoss(
        weight=class_balanced_weight(Counter(ytr.tolist()), n_classes=2, device=str(device)))

    def _feat(Fpre_t, Fpost_t):
        """Project cached features and combine both dates for paired classifiers."""
        f_pre = proj(Fpre_t)
        f_post = proj(Fpost_t) if (concept.is_paired and Fpost_t is not None) else None
        return _combine(f_pre, f_post)

    # Keep cached features on CPU and transfer one training batch at a time.
    Fpre_tr_t = torch.from_numpy(Fpre_tr)  # f16 CPU
    Fpost_tr_t = torch.from_numpy(Fpost_tr) if Fpost_tr is not None else None
    ytr_t = torch.from_numpy(ytr)

    def _va_probabilities():
        """Predict validation probabilities without updating model weights."""
        preds = []
        for i in range(0, len(yva), 4096):
            fpre = torch.from_numpy(Fpre_va[i:i + 4096]).float().to(device)
            fpost = (torch.from_numpy(Fpost_va[i:i + 4096]).float().to(device)
                     if Fpost_va is not None else None)
            preds.append(head(_feat(fpre, fpost)).softmax(-1).cpu())
        return torch.cat(preds).numpy()

    best_f1, best_state = -1.0, None
    n = len(ytr_t)
    bs = args.batch_size_s2
    for ep in range(args.epochs_s2):
        proj.train()
        head.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            fpre = Fpre_tr_t[idx].float().to(device)
            fpost = Fpost_tr_t[idx].float().to(device) if Fpost_tr_t is not None else None
            logits = head(_feat(fpre, fpost))
            crit(logits, ytr_t[idx].to(device)).backward()
            opt.step()
        sched.step()
        proj.eval()
        head.eval()
        with torch.no_grad():
            pred = _va_probabilities().argmax(-1)
        m = f1_metrics(yva, pred)
        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            best_state = tuple({k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
                               for m in (proj, head))
    # Keep the epoch with the best validation macro-F1.
    if best_state is not None:
        proj.load_state_dict(best_state[0])
        head.load_state_dict(best_state[1])
    print(f"[s2 {name}] best val macro-F1={best_f1:.4f} (n_tr={n})", flush=True)
    with torch.no_grad():
        scores = _va_probabilities()[:, 1]
    return {"macro_f1": best_f1, "n_train": int(n)}, scores


def reset_classifier(model, name: str):
    """Re-initialize a classifier's projection and head so that it trains from scratch."""
    for module in (model.concept_proj[name], model.heads[name]):
        for layer in module.modules():
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()


def stage2(args, model, device, run_dir: Path):
    """Train the blocked-road classifier from a fresh start on features of the fixed encoder."""
    model.freeze_backbone()
    (concept,) = model.concepts
    excluded = read_unreadable_scenes(args.unreadable_scenes) if args.unreadable_scenes else frozenset()
    cached = {}
    for split in ("train", "val"):
        rows = load_concept_supervision(labels_dir=args.labels_dir, split_csv=args.split_csv,
                                        roads_csv=args.roads_csv, split=split,
                                        concepts=[concept])[concept.name]
        # Only the listed unreadable validation scenes are skipped.
        if split == "val":
            rows = [r for r in rows if r.chip_id not in excluded]
        Fpre, Fpost, y, ordered_rows = _precompute_pooled(model, rows, concept, device, IMAGE_SIZE)
        if split == "val":
            validation_rows = ordered_rows
        counts = np.bincount(y, minlength=2)
        if counts.min() == 0:
            raise SystemExit(f"[stage2] {concept.name}: {split} labels need both classes, "
                             f"got counts {counts.tolist()}")
        cached[split] = (Fpre, Fpost, y)
    print(
        f"[s2 precompute] {concept.name}: tr={cached['train'][0].shape} "
        f"pos={int(cached['train'][2].sum())} va={len(cached['val'][2])} "
        f"paired={concept.is_paired}",
        flush=True)
    reset_classifier(model, concept.name)
    result, scores = _train_concept(model, concept, *cached["train"], *cached["val"], device, args)
    torch.save({"model_state_dict": model.state_dict(), "config": {"image_size": IMAGE_SIZE}},
               run_dir / "base.pth")
    # Scores of the selected checkpoint, for threshold calibration.
    units = [dict(chip_id=r.chip_id, entity_id=r.entity_id, label=r.label, score=float(s))
             for r, s in zip(validation_rows, scores, strict=True)]
    (run_dir / "validation_road_scores.json").write_text(json.dumps(units, indent=2) + "\n")
    (run_dir / "stage2_results.json").write_text(json.dumps({concept.name: result}, indent=2))
    print(f"[stage2] done -> {run_dir / 'base.pth'}", flush=True)


def build_args():
    """Parse the command line and check the inputs that the chosen stages need."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default="all", choices=["all", "1", "2"])
    ap.add_argument("--output", required=True,
                    help="folder for the checkpoints, validation scores and config.json")
    ap.add_argument("--split-csv", required=True,
                    help="split file with image and road-mask paths: assets/metadata/split.csv")
    ap.add_argument("--unreadable-scenes", help="validation scenes to skip: "
                    "assets/metadata/unreadable_scenes.json")
    # Stage 1 inputs.
    ap.add_argument("--building-entities-dir",
                    help="reference building entity JSONs, such as data/building_entities")
    ap.add_argument("--detector-checkpoint",
                    help="building detector to start from, such as assets/checkpoints/detector.pth")
    # Stage 2 inputs.
    ap.add_argument("--labels-dir", help="label folder with per_entity/{train,val}/road_blocked.csv")
    ap.add_argument("--roads-csv", help="reference road boxes, such as data/roads.csv")
    ap.add_argument("--stage1-checkpoint", help="Stage 1 checkpoint that --stage 2 starts from")
    # Stage 1 settings; --weight-decay and --seed also apply to Stage 2.
    ap.add_argument("--num-iter", type=int, default=40_000)
    ap.add_argument("--warmup-iter", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--backbone-lr", type=float, default=5e-6)
    ap.add_argument("--head-lr", type=float, default=2e-4)
    ap.add_argument("--w-road", type=float, default=0.2)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--amp", choices=["off", "bf16"], default="bf16")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--max-val-samples", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    # Stage 2 settings.
    ap.add_argument("--epochs-s2", type=int, default=30)
    ap.add_argument("--batch-size-s2", type=int, default=256)
    ap.add_argument("--head-lr-s2", type=float, default=1e-3)
    args = ap.parse_args()
    stage1_inputs = ["building_entities_dir", "detector_checkpoint"]
    stage2_inputs = ["labels_dir", "roads_csv"]
    needed = {
        "1": stage1_inputs,
        "2": stage2_inputs + ["stage1_checkpoint"],
        "all": stage1_inputs + stage2_inputs
    }[args.stage]
    missing = ["--" + name.replace("_", "-") for name in needed if not getattr(args, name)]
    if missing:
        ap.error(f"--stage {args.stage} requires {' '.join(missing)}")
    if args.stage1_checkpoint and args.stage != "2":
        ap.error("--stage1-checkpoint is only used with --stage 2")
    return args


def main():
    """Run the requested entity-extraction and classifier training stages."""
    args = build_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.output)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))

    # The default model has one MLP classifier, for blocked roads, which Stage 2 trains.
    # siren.concepts.train_damage trains the damage and destruction heads, and
    # siren.concepts.train the residential and vegetation classifiers.
    # Every encoder weight comes from detector.pth (Stage 1, checked in load_warmstart) or from
    # the Stage 1 checkpoint (Stage 2), so the Satlas weights are not downloaded.
    model = SharedPerceptionModel(pretrained_backbone=False).to(device)
    model.verify_backbone_channels(device)

    if args.stage in ("all", "1"):
        load_warmstart(model, args.detector_checkpoint)
        stage1(args, model, device, run_dir)
    if args.stage in ("all", "2"):
        # Stage 2 starts from the weights of the last Stage 1 step.
        path = args.stage1_checkpoint if args.stage == "2" else run_dir / "stage1_last.pth"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        del checkpoint
        print(f"[stage2] loaded weights from {path}", flush=True)
        stage2(args, model, device, run_dir)


if __name__ == "__main__":
    main()
