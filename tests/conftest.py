"""Shared test fixture: a small invented scene in the export format, for query tests."""
import pytest


def make_example_scene():
    """Three buildings along y = 0 and two roads, with classifier scores."""
    buildings = []
    # 20 x 20 pixel boxes along y = 0; area_px is the mask area used by the "large" rule.
    for entity_id, x, damage, destroyed, residential, vegetation, area in (
        ('b1', 0, .9, .9, .9, .9, 2500),
        ('b2', 200, .8, .1, .9, .1, 1000),
        ('b3', 400, .1, .1, .1, .1, 1000),
    ):
        buildings.append({
            'id': entity_id,
            'geometry': {'bbox_xyxy': [x, 0, x+20, 20], 'area_px': area},
            'attributes': {'is_damaged_score': damage, 'is_destroyed_score': destroyed,
                           'is_residential_score': residential,
                           'surrounded_by_vegetation_score': vegetation},
        })
    # r2 is blocked for road access (0.45) but below the counting threshold (0.83).
    roads = [{'id': 'r1', 'geometry': {'centerline_xy': [[0, 30], [100, 30]]},
              'attributes': {'is_road_blocked_score': .1}},
             {'id': 'r2', 'geometry': {'centerline_xy': [[380, 40], [440, 40]]},
              'attributes': {'is_road_blocked_score': .6}}]
    return {'chip_id': 'synthetic', 'entities': {'buildings': buildings, 'roads': roads}}


@pytest.fixture
def example_scene():
    """A fresh copy of the invented scene for each test."""
    return make_example_scene()
