"""Check the counting programs: thresholds, distance and area rules, incomplete exports."""
import copy

import numpy as np
import pytest

from siren.settings import QUERY_THRESHOLDS
from siren.programs.engine import FORMAT, execute
from siren.programs.recipes import COUNT_QUERIES, count_program, paper_example


def count(query, record):
    return execute(count_program(query), record)["value"]


def kept_ids(record, kind, concept):
    program = {"format": FORMAT, "steps": [
        {"id": "all", "op": "select", "kind": kind},
        {"id": "kept", "op": "filter", "source": "all", "concept": concept, "negate": False},
        {"id": "answer", "op": "ids", "source": "kept"}], "output": "answer"}
    return execute(program, record)["value"]


def float32_around(threshold):
    """The float32 scores just below and just at or above threshold, as classifiers store them."""
    nearest = np.float32(threshold)
    if float(nearest) < threshold:
        return float(nearest), float(np.nextafter(nearest, np.float32(1)))
    return float(np.nextafter(nearest, np.float32(0))), float(nearest)


def building(id="b", damage=.8, destroyed=.8, residential=.8, veg=.8, x=0, area=2000):
    return {"id": id, "geometry": {"bbox_xyxy": [x, 0, x+10, 10], "area_px": area},
            "attributes": {"is_damaged_score": damage, "is_destroyed_score": destroyed,
                           "is_residential_score": residential, "surrounded_by_vegetation_score": veg}}


def road(id="r", blocked=.1, **geometry):
    return {"id": id, "geometry": geometry, "attributes": {"is_road_blocked_score": blocked}}


def scene(*buildings, roads=None):
    return {"chip_id": "scene", "entities": {"buildings": list(buildings), "roads": roads or []}}


def test_residential_and_vegetation_require_destruction():
    record = scene(building(destroyed=.1))
    assert count("destroyed_residential_buildings", record) == 0
    assert count("destroyed_buildings_surrounded_by_vegetation", record) == 0
    with pytest.raises(ValueError, match="Unknown count query"):
        count_program("unknown_query")


def test_concept_only_query_does_not_require_irrelevant_geometry():
    b = building()
    del b["geometry"]
    assert count("destroyed_residential_buildings", scene(b)) == 1


@pytest.mark.parametrize("distance,expected", [(49.99, 0), (50, 0), (50.01, 1), (100, 1)])
def test_single_inclusive_access_radius(distance, expected):
    r = road(bbox_xyxy=[10+distance, 0, 11+distance, 10])
    assert count("damaged_buildings_without_clear_road", scene(building(), roads=[r])) == expected


@pytest.mark.parametrize("gap,expected", [(49.99, 0), (50, 0), (50.01, 1)])
def test_single_inclusive_isolation_radius(gap, expected):
    neighbor = building("intact", damage=.1, destroyed=.1, x=10+gap)
    assert count("destroyed_buildings_without_intact_neighbor", scene(building(), neighbor)) == expected


def test_road_centerline_not_bbox_and_access_threshold():
    r = road(bbox_xyxy=[0, 0, 200, 200], centerline_xy=[[100, 100], [200, 200]])
    record = scene(building(), roads=[r])
    assert count("damaged_buildings_without_clear_road", record) == 1
    r["geometry"]["centerline_xy"] = [[0, 0], [10, 10]]
    assert count("damaged_buildings_without_clear_road", record) == 0
    r["attributes"]["is_road_blocked_score"] = QUERY_THRESHOLDS["is_road_blocked"]
    assert count("damaged_buildings_without_clear_road", record) == 1


def test_building_to_road_distance_uses_centerline_not_box():
    r = road(bbox_xyxy=[0, 0, 200, 200], centerline_xy=[[0, 60], [100, 60]])
    result = execute(paper_example("distance", entity_id="b"), scene(building(), roads=[r]),
                     context={"pixel_size_m": .8})
    # The box overlaps the building; the centerline is 50 px (40 m) away.
    assert result["value"] == pytest.approx(40) and result["unit"] == "m"


@pytest.mark.parametrize("score,clear", [
    (float32_around(.45)[0], True),
    (float32_around(.45)[1], False),
    (QUERY_THRESHOLDS["is_road_blocked"], False),
])
def test_clear_road_boundary_is_045(score, clear):
    roads = [road(blocked=score, centerline_xy=[[0, 20], [10, 20]])]
    assert kept_ids(scene(roads=roads), "roads", "clear") == (["r"] if clear else [])
    assert count("damaged_buildings_without_clear_road", scene(building(), roads=roads)) == (0 if clear else 1)


@pytest.mark.parametrize("score,expected", [
    (.6, 0),
    (float32_around(.83)[0], 0),
    (.83, 1),
    (float32_around(.83)[1], 1),
    (1.0, 1),
])
def test_road_counted_as_blocked_from_083(score, expected):
    roads = [road(blocked=score, centerline_xy=[[0, 0], [10, 0]])]
    assert count("blocked_roads", scene(roads=roads)) == expected


def test_intact_means_below_both_damage_thresholds():
    (damage_below, damage_at) = float32_around(.55)
    (destroyed_below, destroyed_at) = float32_around(.60)
    record = scene(building("a", damage=damage_below, destroyed=destroyed_below),
                   building("b", damage=damage_at, destroyed=.1),
                   building("c", damage=.1, destroyed=destroyed_at),
                   building("d", damage=.55, destroyed=.60),
                   building("e", damage=.1, destroyed=.1))
    assert kept_ids(record, "buildings", "intact") == ["a", "e"]


def test_intact_requires_both_negative():
    destroyed_but_low_damage = building(damage=.1)
    record = scene(destroyed_but_low_damage)
    assert count("destroyed_buildings_without_intact_neighbor", record) == 1
    record["entities"]["buildings"].append(building("intact", damage=.1, destroyed=.1, x=60))
    assert count("destroyed_buildings_without_intact_neighbor", record) == 0


@pytest.mark.parametrize("area,expected", [(1999, 0), (1999.999, 0), (2000, 1), (2001, 1)])
def test_area_uses_exported_mask_area(area, expected):
    b = building(area=area)
    b["attributes"]["area_px"] = 10000  # Must not override the exported mask area.
    b["geometry"]["bbox_xyxy"] = [0, 0, 100, 100]  # Nor may the box area.
    assert count("large_destroyed_buildings", scene(b)) == expected


def test_area_must_come_from_geometry():
    b = building()
    del b["geometry"]
    b["area_px"] = 5000
    with pytest.raises(ValueError, match="geometry.area_px"):
        count("large_destroyed_buildings", scene(b))


@pytest.mark.parametrize("query", COUNT_QUERIES)
def test_explicit_empty_scene_is_zero(query):
    assert count(query, scene()) == 0


@pytest.mark.parametrize("mutation", ["missing_group", "missing_score", "duplicate", "nan", "missing_area"])
def test_incomplete_exports_are_not_zero(mutation):
    record = scene(building())
    if mutation == "missing_group":
        del record["entities"]["buildings"]
    elif mutation == "missing_score":
        del record["entities"]["buildings"][0]["attributes"]["is_destroyed_score"]
    elif mutation == "duplicate":
        record["entities"]["buildings"].append(copy.deepcopy(record["entities"]["buildings"][0]))
    elif mutation == "nan":
        record["entities"]["buildings"][0]["attributes"]["is_destroyed_score"] = float("nan")
    else:
        del record["entities"]["buildings"][0]["geometry"]["area_px"]
    with pytest.raises(ValueError):
        count("large_destroyed_buildings", record)


@pytest.mark.parametrize("bad_entity", [None, [], "building"])
def test_malformed_entity_rows_are_rejected(bad_entity):
    record = scene()
    record["entities"]["buildings"] = [bad_entity]
    with pytest.raises(ValueError, match="Malformed entity"):
        count("destroyed_residential_buildings", record)


def test_road_queries_require_roads():
    record = scene(building())
    del record["entities"]["roads"]
    with pytest.raises(ValueError, match="roads"):
        count("damaged_buildings_without_clear_road", record)


def test_vegetation_concept_name_in_schema():
    import json
    from siren.language.contract import response_schema
    record = scene(building("yes", veg=.9), building("no", veg=.1))
    assert kept_ids(record, "buildings", "surrounded_by_vegetation") == ["yes"]
    assert '"surrounded_by_vegetation"' in json.dumps(response_schema())


@pytest.mark.parametrize('polygon', [None, [[[0, 0], [10, 10], [10, 0], [0, 10]]]])
def test_learning_geometry_matches_query_geometry(polygon):
    from siren.evaluation.learning import counts
    entities = [building('destroyed', x=0), building('intact', x=60, damage=.1, destroyed=.1)]
    if polygon is not None:
        entities[0]['geometry']['polygon_xy'] = polygon
    result = counts(entities, [.8, .1], [.8, .1], .55, .60)
    record = scene(*entities)
    for question, value in result.items():
        assert count(question, record) == value


def test_learning_rejects_nonfinite_geometry():
    from siren.evaluation.learning import counts
    entity = building()
    entity['geometry']['polygon_xy'] = [[[0, 0], [float('nan'), 1], [1, 0]]]
    with pytest.raises(ValueError, match='Invalid geometry'):
        counts([entity], [.8], [.8], .55, .60)
