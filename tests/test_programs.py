"""Check operation semantics, invalid programs and the paper's example programs."""
import pytest
from siren.programs.engine import FORMAT, execute, validate
from siren.programs.recipes import building_fraction, count_program


def scene(scores):
    return {'entities': {'buildings': [{'id': str(i), 'attributes': {'is_damaged_score': s}} for i, s in enumerate(scores)]}}


def test_fraction_and_empty_denominator():
    assert execute(building_fraction(), scene([.8] * 12 + [.1] * 28))['value'] == .3
    assert execute(building_fraction(), scene([]))['status'] == 'undefined'


@pytest.mark.parametrize('mutation', ['unknown', 'forward', 'extra', 'kind', 'negate'])
def test_invalid_program_before_empty_scene(mutation):
    p = building_fraction()
    if mutation == 'unknown': p['steps'][1]['op'] = 'exec'
    if mutation == 'forward': p['steps'][1]['source'] = 'answer'
    if mutation == 'extra': p['steps'][1]['code'] = 'import os'
    if mutation == 'kind': p['steps'][0]['kind'] = 'people'
    if mutation == 'negate': p['steps'][1]['negate'] = 'false'
    with pytest.raises(ValueError): execute(p, scene([]))


def test_missing_entities_or_scores_are_errors_not_zero():
    with pytest.raises(ValueError): execute(building_fraction(), {})
    with pytest.raises(ValueError): execute(building_fraction(), scene([None]))


def test_density_requires_measured_area_and_resolution():
    p = building_fraction()
    p['steps'][-1] = {'id': 'answer', 'op': 'count_density', 'source': 'selected'}
    with pytest.raises(ValueError): execute(p, scene([.9]))
    assert execute(p, scene([.9]), context={'valid_area_px': 10000, 'pixel_size_m': 2})['value'] == 25


def test_subset_enforced():
    p = building_fraction()
    p['steps'][-1].update(source='all', total='selected')
    with pytest.raises(ValueError): execute(p, scene([.1, .9]))


def test_count_programs_synthetic_answers():
    s = {'entities': {'buildings': [], 'roads': []}}
    for i in range(8):
        s['entities']['buildings'].append({'id': str(i), 'geometry': {'area_px': 1000*(i+1), 'bbox_xyxy':[100*i,0,100*i+20,20]}, 'attributes': {
            'is_damaged_score': .8 if i%2 else .1, 'is_destroyed_score': .8 if i%3 else .1,
            'is_residential_score': .8 if i%4 else .1, 'surrounded_by_vegetation_score': .8 if i%5 else .1}})
    # Expected answers on this synthetic scene (no roads; boxes 80 pixels apart).
    expected = {'damaged_buildings_without_clear_road': 4, 'destroyed_buildings_without_intact_neighbor': 5, 'large_destroyed_buildings': 5,
                'destroyed_residential_buildings': 4, 'destroyed_buildings_surrounded_by_vegetation': 4}
    for q, answer in expected.items():
        assert execute(count_program(q), s)['value'] == answer


def test_unused_valid_branch_does_not_require_missing_geometry():
    p = building_fraction()
    p['steps'].insert(1, {'id': 'unused', 'op': 'relate', 'source': 'all', 'target': 'all',
                          'distance_px': 0, 'policy': 'footprint', 'mode': 'any'})
    r = execute(p, scene([.9, .1]))
    assert r['value'] == .5 and r['unused_steps'] == ['unused']
    p['steps'][1]['op'] = 'unknown'
    with pytest.raises(ValueError):
        execute(p, scene([]))


def spatial_scene():
    buildings = []
    for name, x, destroyed in [('a', 0, .9), ('b', 80, .1), ('c', 90, .1)]:
        buildings.append({'id': name, 'geometry': {'bbox_xyxy': [x, 0, x+10, 10]},
                          'attributes': {'is_destroyed_score': destroyed, 'is_damaged_score': .9}})
    roads = [
        {'id': 'blocked', 'geometry': {'centerline_xy': [[0, 20], [10, 20]]},
         'attributes': {'is_road_blocked_score': .9}},
        {'id': 'clear', 'geometry': {'centerline_xy': [[0, 30], [100, 30]]},
         'attributes': {'is_road_blocked_score': .1}},
    ]
    return {'entities': {'buildings': buildings, 'roads': roads}}


def test_six_paper_examples_validate_and_run():
    from siren.programs.recipes import paper_example
    from siren.language.compiler import execute_compiled, validate_response, system_prompt
    import json

    context = {'image_width_px': 100, 'valid_area_px': 10000, 'pixel_size_m': 2}
    expected = {'density': 25, 'fraction': 1/3, 'threshold': False,
                'comparison': 'greater', 'nearest': ['a'], 'distance': 40}
    for kind, answer in expected.items():
        program = paper_example(kind, entity_id='a')
        response = {'program': program, 'reason': ''}
        validate_response(json.dumps(response))
        result = execute_compiled(response, spatial_scene(), context=context)
        assert result['value'] == answer
    assert 'Use nearest' in system_prompt()


def test_nearest_uses_geometry_and_preserves_ties():
    from siren.programs.recipes import paper_example
    s = spatial_scene()
    s['entities']['buildings'][1]['geometry']['bbox_xyxy'] = [0, 0, 10, 10]
    # A broad box must not replace the road centerline.
    s['entities']['roads'][0]['geometry']['bbox_xyxy'] = [0, 0, 100, 100]
    assert execute(paper_example('nearest'), s)['value'] == ['a', 'b']


def test_distance_requires_scale_and_reports_empty_target():
    from siren.programs.recipes import paper_example
    p = paper_example('distance', entity_id='a')
    s = spatial_scene()
    with pytest.raises(ValueError):
        execute(p, s)
    s['entities']['roads'] = []
    result = execute(p, s, context={'pixel_size_m': .8})
    assert result['value'] is None and result['status'] == 'undefined'
    with pytest.raises(ValueError, match='absent'):
        execute(paper_example('distance', entity_id='missing'), s)


def test_distance_accepts_an_undamaged_building():
    from siren.programs.recipes import paper_example
    s = spatial_scene()
    s['entities']['buildings'][0]['attributes']['is_damaged_score'] = .1
    result = execute(paper_example('distance', entity_id='a'), s,
                     context={'pixel_size_m': 2})
    assert result['value'] == 40


def test_comparison_empty_half_and_midpoint():
    from siren.programs.recipes import paper_example
    s = spatial_scene()
    p = paper_example('comparison')
    result = execute(p, s, context={'image_width_px': 10})
    assert result['status'] == 'undefined'
    assert next(step['value'] for step in result['trace'] if step['id'] == 'left') == []


def test_threshold_is_inclusive_and_preserves_undefined():
    from siren.programs.recipes import paper_example
    s = spatial_scene()
    p = paper_example('threshold')
    p['steps'][-1]['value'] = 1
    assert execute(p, s)['value'] is True
    p['steps'][-1]['comparison'] = 'gt'
    assert execute(p, s)['value'] is False
    p = building_fraction()
    p['steps'].append(dict(id='check', op='threshold', source='answer', value=.5, comparison='ge'))
    p['output'] = 'check'
    assert execute(p, scene([]))['status'] == 'undefined'


def program(*steps):
    return {'format': FORMAT, 'steps': list(steps), 'output': 'answer'}


SELECT = dict(id='all', op='select', kind='buildings')
DESTROYED = dict(id='destroyed', op='filter', source='all', concept='destroyed', negate=False)


def destroyed_buildings(*positions):
    return {'entities': {'buildings': [
        {'id': name, 'geometry': {'bbox_xyxy': [x, 0, x+10, 10]}, 'attributes': {'is_destroyed_score': .9}}
        for name, x in positions]}}


@pytest.mark.parametrize('mode,expected', [('any', ['a', 'b']), ('none', ['far'])])
def test_relate_never_matches_an_entity_with_itself(mode, expected):
    s = destroyed_buildings(('a', 0), ('b', 10), ('far', 500))
    relate = dict(id='near', op='relate', source='destroyed', target='destroyed', distance_px=.5,
                  policy='footprint', mode=mode)
    answer = dict(id='answer', op='ids', source='near')
    assert execute(program(SELECT, DESTROYED, relate, answer), s)['value'] == expected
    # A second select returns the same scene rows, so identity still marks the same entity.
    again = dict(id='again', op='select', kind='buildings')
    p = program(SELECT, DESTROYED, again, {**relate, 'target': 'again'}, answer)
    assert execute(p, s)['value'] == expected


def test_nearest_and_distance_skip_the_entity_itself():
    s = destroyed_buildings(('a', 0), ('b', 30), ('c', 300))
    chosen = dict(id='chosen', op='entity', source='all', entity_id='a')
    nearest = dict(id='near', op='nearest', source='destroyed', target='chosen',
                   policy='footprint')
    answer = dict(id='answer', op='ids', source='near')
    assert execute(program(SELECT, DESTROYED, chosen, nearest, answer), s)['value'] == ['b']
    distance = dict(id='answer', op='distance', source='chosen', target='destroyed',
                    policy='footprint')
    result = execute(program(SELECT, DESTROYED, chosen, distance), s,
                     context={'pixel_size_m': .5})
    assert result['value'] == 10 and result['unit'] == 'm'
    # An entity alone has no other target.
    alone = program(SELECT, chosen, {**distance, 'target': 'chosen'})
    assert execute(alone, s, context={'pixel_size_m': .5})['status'] == 'undefined'
    alone = program(SELECT, chosen, {**nearest, 'source': 'chosen'}, answer)
    assert execute(alone, s)['value'] == []


def test_density_is_whole_image_only():
    from siren.language.contract import DESCRIPTIONS
    half = dict(id='left', op='half', source='all', side='left')
    p = program(SELECT, half, {**DESTROYED, 'source': 'left'},
                dict(id='answer', op='count_density', source='destroyed'))
    with pytest.raises(ValueError, match='whole image'):
        execute(p, scene([]))
    assert 'whole image only' in DESCRIPTIONS['count_density']


def test_extreme_selects_building_areas_and_keeps_ties():
    s = {'entities': {'buildings': [{'id': name, 'geometry': {'area_px': area}}
                                    for name, area in [('a', 3000), ('b', 3000), ('c', 1000)]]}}
    for selection, expected in [('largest', ['a', 'b']), ('smallest', ['c'])]:
        p = program(SELECT, dict(id='pick', op='extreme', source='all', selection=selection),
                    dict(id='answer', op='ids', source='pick'))
        assert execute(p, s)['value'] == expected
    p['steps'][0] = dict(id='all', op='select', kind='roads')
    with pytest.raises(ValueError, match='building footprint'):
        validate(p)


def test_compare_units_checked_before_execution():
    p = building_fraction()
    p['steps'] += [dict(id='number', op='count', source='selected'),
                   dict(id='check', op='compare', source='answer', target='number')]
    p['output'] = 'check'
    with pytest.raises(ValueError, match='matching units'):
        validate(p)


def test_missing_context_names_the_key():
    from siren.programs.recipes import paper_example
    with pytest.raises(ValueError, match='image_width_px'):
        execute(paper_example('comparison'), spatial_scene())
    with pytest.raises(ValueError, match='pixel_size_m'):
        execute(paper_example('distance', entity_id='a'), spatial_scene())



def test_validate_rejects_non_string_field_values():
    program = {'format': FORMAT, 'output': 'n', 'steps': [
        {'id': 'b', 'op': 'select', 'kind': ['buildings']},
        {'id': 'n', 'op': 'count', 'source': 'b'}]}
    with pytest.raises(ValueError, match='kind must be a string'):
        validate(program)
