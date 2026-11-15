"""Building and road extraction F1, used to monitor Stage 1 and to choose the road threshold on
validation data. Like Stage 1 training, it uses scenes with at least one annotated building.
Scene-count evaluation is in siren.evaluation.counts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from siren.metrics import precision_recall_f1
from siren.perception.data import MultiTaskChipDataset, collate_multitask
from siren.perception.loading import load_checkpoint
from siren.perception.model import SharedPerceptionModel
from siren.segmentation.building.utils import boxes_iou, greedy_match_count
from siren.settings import IMAGE_SIZE, ROADS


@torch.no_grad()
def building_f1(model: SharedPerceptionModel,
                loader,
                device,
                *,
                score_thresh: float = 0.05,
                iou_thresh: float = 0.5) -> dict:
    """Building detection F1: one-to-one greedy box matching at IoU >= 0.5."""
    model.eval()
    tp = fp = fn = 0
    for batch in loader:
        pre = batch["pre"].to(device)
        dets = model.det(pre)  # one dict of boxes, labels, scores and masks per image
        for det, gt in zip(dets, batch["targets"]):
            keep = det["scores"] >= score_thresh
            pred = det["boxes"][keep].detach().cpu().numpy().astype(np.float32)
            gt_boxes = gt["boxes"].detach().cpu().numpy().astype(np.float32)
            iou = boxes_iou(pred, gt_boxes) if len(pred) and len(gt_boxes) \
                else np.zeros((len(pred), len(gt_boxes)), np.float32)
            m = greedy_match_count(iou, iou_thresh)
            tp += m
            fp += len(pred) - m
            fn += len(gt_boxes) - m
    p, r, f1 = precision_recall_f1(tp, fp, fn)
    return {"f1": f1, "precision": p, "recall": r, "tp": tp, "fp": fp, "fn": fn}


ROAD_THRESH_GRID = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]  # tried by --road-sweep


@torch.no_grad()
def road_f1(model: SharedPerceptionModel,
            loader,
            device,
            *,
            thresh: float = ROADS.road_thresh,
            grid: list[float] | None = None) -> dict:
    """Road-mask F1 over the pixels of all scenes with road labels; Stage 1 logs it at the
    default threshold.

    With grid, sweep the thresholds and return the best one (use validation data only)."""
    model.eval()
    thrs = grid if grid is not None else [thresh]
    tp = {t: 0 for t in thrs}
    fp = {t: 0 for t in thrs}
    fn = {t: 0 for t in thrs}
    for batch in loader:
        pre = batch["pre"].to(device)
        valid = batch["road_valid"]
        prob = torch.sigmoid(model.road_logits(pre).squeeze(1))  # (B,H,W)
        gt = batch["road_semantic"].to(device).bool()
        for i in range(prob.shape[0]):
            if not bool(valid[i]):
                continue
            gi = gt[i]
            for t in thrs:  # pixel counts pooled over scenes with road labels
                pi = prob[i] >= t
                tp[t] += int((pi & gi).sum())
                fp[t] += int((pi & ~gi).sum())
                fn[t] += int((~pi & gi).sum())
    per_thr = {}
    for t in thrs:
        p, r, f1 = precision_recall_f1(tp[t], fp[t], fn[t])
        per_thr[t] = {"f1": f1, "precision": p, "recall": r}
    best_t = max(per_thr, key=lambda t: per_thr[t]["f1"])
    out = dict(per_thr[best_t])
    out["thresh"] = best_t
    if grid is not None:
        out["sweep"] = {str(t): per_thr[t]["f1"] for t in thrs}
    return out


def main():
    """Load a checkpoint and evaluate building and road extraction."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True,
                    help="base.pth or a Stage 1 checkpoint (stage1_last.pth)")
    ap.add_argument("--building-score-threshold",
                    type=float,
                    default=0.5,
                    help="detection score cutoff for building F1 "
                    "(Stage 1 logs building F1 at 0.05)")
    ap.add_argument("--split-csv", required=True,
                    help="split file with image and road-mask paths: assets/metadata/split.csv")
    ap.add_argument("--building-entities-dir", required=True,
                    help="reference building entity JSONs, such as data/building_entities")
    ap.add_argument("--unreadable-scenes", help="validation scenes to skip: "
                    "assets/metadata/unreadable_scenes.json")
    ap.add_argument("--output", required=True, help="output folder")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument(
        "--road-sweep",
        action="store_true",
        help="report road F1 over a grid of thresholds and pick the best (validation split only)")
    ap.add_argument("--road-thresh",
                    type=float,
                    default=ROADS.road_thresh,
                    help="road probability threshold (default 0.80, the paper setting)")
    args = ap.parse_args()
    if args.road_sweep and args.split == "test":
        raise SystemExit("--road-sweep chooses a threshold, so it cannot use the test split. "
                         "Run it with --split val and pass the result to --road-thresh.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(args.checkpoint, extraction=ROADS, device=device).model

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = MultiTaskChipDataset(split_csv=args.split_csv,
                                   building_entities_dir=args.building_entities_dir,
                                   split=args.split,
                                   image_size=IMAGE_SIZE,
                                   unreadable_scenes=args.unreadable_scenes)
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers,
                        collate_fn=collate_multitask)
    seg = {
        "building": building_f1(model, loader, device, score_thresh=args.building_score_threshold),
        "building_score_thresh": args.building_score_threshold,
        "road": road_f1(model,
                        loader,
                        device,
                        thresh=args.road_thresh,
                        grid=ROAD_THRESH_GRID if args.road_sweep else None)
    }
    print(json.dumps(seg, indent=2))
    (out_dir / f"seg_metrics_{args.split}.json").write_text(json.dumps(seg, indent=2))


if __name__ == "__main__":
    main()
