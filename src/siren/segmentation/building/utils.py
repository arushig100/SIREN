"""Building mask polygons, duplicate-detection merging and predicted/reference box matching."""
from __future__ import annotations

import numpy as np


def mask_to_polygons(mask: np.ndarray, *, simplify_eps: float, min_area: float) -> list[list[list[float]]]:
    """Outer contours of a binary mask as [x, y] rings in mask pixels, most vertices first.

    Contours under min_area are skipped, the rest are simplified with tolerance simplify_eps,
    and rings left with fewer than 3 vertices are dropped."""
    import cv2

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[list[float]]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        if simplify_eps > 0:
            contour = cv2.approxPolyDP(contour, simplify_eps, True)
        points = contour.reshape(-1, 2)
        if points.shape[0] < 3:
            continue
        polygons.append([[float(x), float(y)] for x, y in points])
    polygons.sort(key=len, reverse=True)
    return polygons


def merge_duplicate_boxes(boxes, scores, *, iou: float, containment: float) -> list[int]:
    """Greedy suppression in stable descending score order; return the kept indices.

    A box is a duplicate of a kept box when their IoU is at least iou or their overlap
    covers at least containment of the smaller box."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    kept: list[int] = []
    for i in order.tolist():
        if kept:
            other = boxes[kept]
            width = np.clip(np.minimum(other[:, 2], boxes[i, 2]) -
                            np.maximum(other[:, 0], boxes[i, 0]), 0, None)
            height = np.clip(np.minimum(other[:, 3], boxes[i, 3]) -
                             np.maximum(other[:, 1], boxes[i, 1]), 0, None)
            overlap = width * height
            if ((overlap / (areas[kept] + areas[i] - overlap + 1e-9) >= iou) |
                    (overlap / (np.minimum(areas[kept], areas[i]) + 1e-9) >= containment)).any():
                continue
        kept.append(i)
    return kept


def boxes_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of (x1, y1, x2, y2) boxes: shape (len(a), len(b))."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ix1 = np.maximum(ax1, bx1[None, :])
    iy1 = np.maximum(ay1, by1[None, :])
    ix2 = np.minimum(ax2, bx2[None, :])
    iy2 = np.minimum(ay2, by2[None, :])
    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b[None, :] - inter
    return (inter / np.maximum(union, 1e-6)).astype(np.float32)


def greedy_match_count(iou: np.ndarray, threshold: float) -> int:
    """Number of one-to-one matches with IoU >= threshold, taking the highest IoU first."""
    pairs = [
        (float(iou[i, j]), i, j)
        for i in range(iou.shape[0])
        for j in range(iou.shape[1])
        if iou[i, j] >= threshold
    ]
    pairs.sort(reverse=True)
    used_i: set[int] = set()
    used_j: set[int] = set()
    matched = 0
    for _, i, j in pairs:
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        matched += 1
    return matched
