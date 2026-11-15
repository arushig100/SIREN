"""Classification and count metrics."""
from __future__ import annotations

import numpy as np


def precision_recall_f1(tp: int, fp: int, fn: int,
                         *, eps: float = 1e-12) -> tuple[float, float, float]:
    """Return precision, recall and F1; empty denominators yield zero."""
    p = tp / max(eps, tp + fp)
    r = tp / max(eps, tp + fn)
    f = (2 * p * r) / max(eps, p + r)
    return p, r, f


def f1_score(tp: int, fp: int, fn: int) -> float:
    """Return F1 from true-positive, false-positive and false-negative counts."""
    return precision_recall_f1(tp, fp, fn)[2]


def binary_f1(tp: int, fp: int, fn: int) -> float:
    """F1 as 2 TP / (2 TP + FP + FN); 0 when there are no positives and no predictions."""
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def macro_f1(y, p) -> float:
    """Mean F1 of classes 0 and 1, from 0/1 labels y and 0/1 predictions p (arrays)."""
    y, p = np.asarray(y), np.asarray(p)
    scores = [binary_f1(int(((p == k) & (y == k)).sum()), int(((p == k) & (y != k)).sum()),
                        int(((p != k) & (y == k)).sum())) for k in (0, 1)]
    return (scores[0] + scores[1]) / 2


def round_count(raw: float) -> int:
    """Round to the nearest integer (Python round: halves go to the even integer), then clip at zero."""
    return max(0, int(round(float(raw))))


def count_mae(references: list[float], predictions: list[int]) -> float:
    """Mean absolute count error; 0 for an empty list."""
    errors = [abs(r - p) for r, p in zip(references, predictions, strict=True)]
    return float(np.mean(errors)) if errors else 0.0
