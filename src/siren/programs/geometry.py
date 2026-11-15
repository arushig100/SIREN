"""Entity shapes and distances for the program engine's spatial operations."""
from __future__ import annotations

import math
from typing import Any

from shapely.geometry import LineString, MultiPolygon, Polygon, box

Point = tuple[float, float]
BBox = tuple[float, float, float, float]


def bbox(entity: dict[str, Any]) -> BBox:
    """Return (x1, y1, x2, y2) from geometry.bbox_xyxy; raise if it is missing or malformed."""
    geom = entity.get("geometry") or {}
    values = geom.get("bbox_xyxy")
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError(f"Entity {entity.get('id')} missing bbox_xyxy")
    x1, y1, x2, y2 = (float(v) for v in values)  # float() rejects nonnumeric values
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        raise ValueError(f"Entity {entity.get('id')} has non-finite bbox_xyxy={values!r}")
    return x1, y1, x2, y2


def polygon_rings(entity: dict[str, Any]) -> list[list[Point]]:
    """Clean polygon rings (each >= 3 coordinate pairs) from geometry.polygon_xy."""
    geom = entity.get("geometry") or {}
    rings = geom.get("polygon_xy") or []
    out: list[list[Point]] = []
    for ring in rings:
        clean = []
        for point in ring:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                clean.append((float(point[0]), float(point[1])))
        if len(clean) >= 3:
            out.append(clean)
    return out


def _finite_geometry(entity):
    """Reject polygon or centerline points with non-finite coordinates."""
    geom = entity.get("geometry") or {}
    for points in [*(geom.get("polygon_xy") or []), geom.get("centerline_xy") or []]:
        for point in points:
            if len(point) < 2 or not all(math.isfinite(float(v)) for v in point[:2]):
                raise ValueError(f"Invalid geometry coordinates for {entity.get('id')}")


def shape(entity, policy):
    """The shapely geometry used for distances under a policy; the box when nothing better exists."""
    _finite_geometry(entity)
    geom = entity.get("geometry") or {}
    if policy == "footprint":
        # The first footprint ring (the exporter lists the ring with the most vertices first).
        # An invalid ring is repaired with buffer(0), which can drop part of a
        # self-intersecting ring.
        rings = geom.get("polygon_xy") or []
        if rings:
            poly = Polygon(rings[0])
            poly = poly if poly.is_valid else poly.buffer(0)
            if not poly.is_empty and poly.geom_type in ("Polygon", "MultiPolygon"):
                return poly
    elif policy == "footprint_or_centerline":
        # All valid footprint rings, else the road centerline, else the box. Invalid rings are
        # skipped, not repaired.
        polys = [Polygon(ring) for ring in polygon_rings(entity)]
        polys = [p for p in polys if not p.is_empty and p.is_valid and p.area > 0]
        if polys:
            return polys[0] if len(polys) == 1 else MultiPolygon(polys)
        line = geom.get("centerline_xy") or []
        if len(line) >= 2:
            return LineString(line)
    else:
        raise ValueError(f"Unknown geometry policy: {policy}")
    return box(*bbox(entity))


def near(a, b, distance, policy):
    """True when the minimum distance between the two geometries is <= distance pixels."""
    value = float(shape(a, policy).distance(shape(b, policy)))
    if not math.isfinite(value):
        raise ValueError("Non-finite geometry distance")
    return value <= distance
