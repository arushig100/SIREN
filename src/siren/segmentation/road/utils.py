"""Clean the road mask, trace its skeleton and split it into road segments.
Each retained segment becomes an entity with a centerline and bounding box."""
from __future__ import annotations

import math
from collections import deque

import numpy as np


def require_skeleton_deps():
    """Import OpenCV, SciPy and scikit-image, with a clear error if one is missing."""
    try:
        import cv2
        from scipy import ndimage as ndi
        from skimage.morphology import skeletonize
    except ImportError as exc:
        raise RuntimeError(
            "Road entity export requires cv2, scipy, and scikit-image. "
            "Install the packages in requirements-model.txt before export."
        ) from exc
    return cv2, ndi, skeletonize


def clean_road_mask(mask: np.ndarray, *, min_area: int, close_radius: int) -> np.ndarray:
    """Close gaps with a disc of radius close_radius, then drop components under min_area pixels."""
    cv2, ndi, _ = require_skeleton_deps()
    out = mask.astype(bool)
    if close_radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_radius + 1, 2 * close_radius + 1))
        out = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    if min_area > 1:
        labels, _ = ndi.label(out)  # 4-connected components
        keep = np.bincount(labels.ravel()) >= min_area
        keep[0] = False
        out = keep[labels]
    return out.astype(bool)


def _neighbors(y: int, x: int, pixels: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """The 8-connected neighbors of (y, x) that are in pixels."""
    out = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            nb = (y + dy, x + dx)
            if nb in pixels:
                out.append(nb)
    return out


def _path_length(path: list[tuple[int, int]]) -> float:
    """Length in pixels of a path of (y, x) points."""
    total = 0.0
    for (y0, x0), (y1, x1) in zip(path, path[1:]):
        total += math.hypot(float(y1 - y0), float(x1 - x0))
    return total


def _component_pixels(start: tuple[int, int], remaining: set[tuple[int, int]]) -> set[tuple[int, int]]:
    """Remove the 8-connected component of start from remaining and return it."""
    q = deque([start])
    comp = {start}
    remaining.remove(start)
    while q:
        y, x = q.popleft()
        for nb in _neighbors(y, x, remaining):
            remaining.remove(nb)
            comp.add(nb)
            q.append(nb)
    return comp


def _trace_from_endpoint(
    endpoint: tuple[int, int],
    pixels: set[tuple[int, int]],
    degree: dict[tuple[int, int], int],
    *,
    max_length_px: float,
) -> tuple[list[tuple[int, int]], float]:
    """Trace a terminal skeleton branch until a graph node or max length."""
    path = [endpoint]
    length = 0.0
    prev = None
    cur = endpoint
    while True:
        nexts = [p for p in _neighbors(cur[0], cur[1], pixels) if p != prev]
        if not nexts:
            break
        nxt = nexts[0]
        length += math.hypot(float(nxt[0] - cur[0]), float(nxt[1] - cur[1]))
        path.append(nxt)
        prev, cur = cur, nxt
        if length >= max_length_px:
            break
        if degree.get(cur, 0) != 2:
            break
    return path, length


def prune_skeleton_spurs(skel: np.ndarray, *, min_spur_length_px: float) -> np.ndarray:
    """Remove branches shorter than min_spur_length_px that join an endpoint to a junction.

    Longer dead-end roads stay."""
    if min_spur_length_px <= 0:
        return skel.astype(bool)
    out = skel.astype(bool).copy()
    changed = True
    while changed:
        changed = False
        ys, xs = np.where(out)
        pixels = set(zip(ys.tolist(), xs.tolist()))
        if not pixels:
            break
        degree = {p: len(_neighbors(p[0], p[1], pixels)) for p in pixels}
        endpoints = [p for p, deg in degree.items() if deg == 1]
        remove: set[tuple[int, int]] = set()
        for endpoint in endpoints:
            if endpoint in remove:
                continue
            path, length = _trace_from_endpoint(
                endpoint,
                pixels,
                degree,
                max_length_px=min_spur_length_px,
            )
            terminal = path[-1]
            if length < min_spur_length_px and degree.get(terminal, 0) >= 3:
                # Drop only the terminal branch; keep the junction so the graph stays connected.
                remove.update(path[:-1])
        if remove:
            yy, xx = zip(*remove)
            out[np.asarray(yy), np.asarray(xx)] = False
            changed = True
    return out


def skeleton_segments(
    mask: np.ndarray,
    *,
    min_length_px: float,
    spur_prune_px: float,
) -> tuple[list[list[tuple[int, int]]], np.ndarray]:
    """Split the mask skeleton into paths between endpoints and junctions; keep long ones.

    Returns the (y, x) pixel paths and the distance to the mask edge."""
    _, ndi, skeletonize = require_skeleton_deps()
    skel = skeletonize(mask.astype(bool))
    # Remove short side branches before splitting the skeleton at junctions.
    skel = prune_skeleton_spurs(skel, min_spur_length_px=spur_prune_px)
    dist = ndi.distance_transform_edt(mask.astype(bool))
    ys, xs = np.where(skel)
    pixels = set(zip(ys.tolist(), xs.tolist()))
    if not pixels:
        return [], dist

    degree = {p: len(_neighbors(p[0], p[1], pixels)) for p in pixels}
    nodes = {p for p, deg in degree.items() if deg != 2}
    visited_edges: set[frozenset[tuple[int, int]]] = set()
    segments: list[list[tuple[int, int]]] = []

    for node in sorted(nodes):
        for nb in _neighbors(node[0], node[1], pixels):
            edge = frozenset((node, nb))
            if edge in visited_edges:
                continue
            path = [node, nb]
            visited_edges.add(edge)
            prev, cur = node, nb
            while cur not in nodes:
                nexts = [p for p in _neighbors(cur[0], cur[1], pixels) if p != prev]
                if not nexts:
                    break
                nxt = nexts[0]
                visited_edges.add(frozenset((cur, nxt)))
                path.append(nxt)
                prev, cur = cur, nxt
            if _path_length(path) >= min_length_px:
                segments.append(path)

    # Closed loops have no endpoint or junction, so the walk above never starts on them.
    # They are traced only when no other segment was found: a scene with any other road
    # segment drops its isolated loops.
    if not segments:
        remaining = set(pixels)
        while remaining:
            comp = _component_pixels(next(iter(remaining)), remaining)
            path = _order_cycle_or_chain(comp)
            if len(path) > 1 and _path_length(path) >= min_length_px:
                segments.append(path)

    return segments, dist


def _order_cycle_or_chain(comp: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """Walk a component from its smallest (y, x) pixel and return the visited path."""
    start = min(comp)
    path = [start]
    prev = None
    cur = start
    for _ in range(len(comp) - 1):
        nexts = [p for p in _neighbors(cur[0], cur[1], comp) if p != prev and p not in path]
        if not nexts:
            break
        nxt = min(nexts)
        path.append(nxt)
        prev, cur = cur, nxt
    return path


def bbox_from_points(points_yx: list[tuple[int, int]], shape: tuple[int, int], pad: float = 0.0) -> list[float]:
    """Padded box (x1, y1, x2, y2) around (y, x) pixels, clipped to the image; x2, y2 exclusive."""
    ys = np.asarray([p[0] for p in points_yx], dtype=np.float32)
    xs = np.asarray([p[1] for p in points_yx], dtype=np.float32)
    height, width = shape
    return [
        float(max(0.0, xs.min() - pad)),
        float(max(0.0, ys.min() - pad)),
        float(min(float(width), xs.max() + 1.0 + pad)),
        float(min(float(height), ys.max() + 1.0 + pad)),
    ]


def make_road_entities(
    road_mask: np.ndarray,
    *,
    min_area: int,
    close_radius: int,
    min_length_px: float,
    spur_prune_px: float,
) -> list[dict]:
    """One road entity per skeleton segment of the cleaned mask; coordinates are in pixels."""
    clean = clean_road_mask(road_mask, min_area=min_area, close_radius=close_radius)
    segments, dist = skeleton_segments(clean, min_length_px=min_length_px, spur_prune_px=spur_prune_px)
    entities = []
    for path in segments:
        length_px = _path_length(path)
        # Road width at each centerline pixel is twice its distance to the mask edge; the box
        # around the centerline is padded by half the mean width, at least 2 pixels.
        widths = [float(dist[y, x] * 2.0) for y, x in path if dist[y, x] > 0]
        mean_width_px = float(np.mean(widths)) if widths else None
        pad = max(2.0, float(mean_width_px or 0.0) / 2.0)
        centerline = [[float(x), float(y)] for y, x in path]
        bbox = bbox_from_points(path, clean.shape, pad=pad)
        entities.append({
            "id": f"road:{len(entities) + 1:06d}",
            "type": "road",
            "geometry": {"bbox_xyxy": bbox, "centerline_xy": centerline},
            "attributes": {"length_px": float(length_px), "mean_width_px": mean_width_px},
        })
    return entities
