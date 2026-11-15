"""Programs for the paper's count queries and spatial examples."""
from siren.settings import NEAR_DISTANCE_PX
from .engine import FORMAT


def building_fraction(concept='damaged'):
    """Fraction of buildings with the concept, for example damaged buildings / all buildings."""
    return {
        'format': FORMAT,
        'steps': [
            {
                'id': 'all',
                'op': 'select',
                'kind': 'buildings'
            },
            {
                'id': 'selected',
                'op': 'filter',
                'source': 'all',
                'concept': concept,
                'negate': False
            },
            {
                'id': 'answer',
                'op': 'fraction',
                'source': 'selected',
                'total': 'all'
            },
        ],
        'output': 'answer'
    }


# Concepts filtered by each of the eight benchmark counting programs, in the paper's order.
COUNT_FILTERS = {
    'damaged_buildings': ['damaged'],
    'destroyed_buildings': ['destroyed'],
    'blocked_roads': ['blocked_for_counting'],
    'damaged_buildings_without_clear_road': ['damaged'],
    'destroyed_buildings_without_intact_neighbor': ['destroyed'],
    'large_destroyed_buildings': ['destroyed', 'large'],
    'destroyed_residential_buildings': ['destroyed', 'residential'],
    'destroyed_buildings_surrounded_by_vegetation': ['destroyed', 'surrounded_by_vegetation'],
}
COUNT_QUERIES = tuple(COUNT_FILTERS)


def count_program(query_id):
    """Build one of the eight benchmark counting programs."""
    if query_id not in COUNT_FILTERS:
        raise ValueError(f'Unknown count query: {query_id!r}')
    steps = [{
        'id': 'all',
        'op': 'select',
        'kind': 'roads' if query_id == 'blocked_roads' else 'buildings'
    }]
    source = 'all'
    for i, concept in enumerate(COUNT_FILTERS[query_id]):
        name = f'filter_{i}'
        steps.append({
            'id': name,
            'op': 'filter',
            'source': source,
            'concept': concept,
            'negate': False
        })
        source = name
    # Remove buildings within NEAR_DISTANCE_PX of a clear road or an intact building.
    if query_id in ('damaged_buildings_without_clear_road', 'destroyed_buildings_without_intact_neighbor'):
        road = query_id == 'damaged_buildings_without_clear_road'
        steps.extend([
            {
                'id': 'targets',
                'op': 'select',
                'kind': 'roads' if road else 'buildings'
            },
            {
                'id': 'eligible',
                'op': 'filter',
                'source': 'targets',
                'concept': 'clear' if road else 'intact',
                'negate': False
            },
            {
                'id': 'related',
                'op': 'relate',
                'source': source,
                'target': 'eligible',
                'distance_px': NEAR_DISTANCE_PX,
                'policy': 'footprint_or_centerline' if road else 'footprint',
                'mode': 'none'
            },
        ])
        source = 'related'
    steps.append({'id': 'answer', 'op': 'count', 'source': source})
    return {'format': FORMAT, 'steps': steps, 'output': 'answer'}


def blocked_roads_near_homes(distance_m=40, pixel_size_m=0.30517578125):
    """Blocked road segments within distance_m of a residential building (Hurricane Ian example).

    The default pixel size is the external analysis grid (312.5 m per 1024 pixels), so 40 m is
    about 131 pixels."""
    if distance_m < 0 or pixel_size_m <= 0:
        raise ValueError('Distance must be nonnegative and pixel size positive')
    steps = [
        dict(id='roads', op='select', kind='roads'),
        dict(id='blocked', op='filter', source='roads', concept='blocked', negate=False),
        dict(id='buildings', op='select', kind='buildings'),
        dict(id='homes', op='filter', source='buildings', concept='residential', negate=False),
        dict(id='nearby',
             op='relate',
             source='blocked',
             target='homes',
             mode='any',
             policy='footprint_or_centerline',
             distance_px=distance_m / pixel_size_m),
        dict(id='answer', op='ids', source='nearby'),
    ]
    return dict(format=FORMAT, steps=steps, output='answer')


def paper_example(kind, *, entity_id=None):
    """Build a paper example program: density, fraction, comparison, threshold, nearest or distance.

    Distance requires a building ID; the threshold example asks for at least 10 destroyed
    buildings. Geometry and scale come from the scene and execution context.
    """
    if kind in ('density', 'fraction', 'threshold'):
        program = building_fraction('destroyed')
        if kind == 'density':
            program['steps'][-1] = dict(id='answer', op='count_density', source='selected')
        elif kind == 'threshold':
            program['steps'][-1] = dict(id='number', op='count', source='selected')
            program['steps'].append(
                dict(id='answer', op='threshold', source='number', value=10, comparison='ge'))
        return program
    steps = [dict(id='buildings', op='select', kind='buildings')]
    if kind == 'comparison':
        for side in ('left', 'right'):
            steps.extend([
                dict(id=side, op='half', source='buildings', side=side),
                dict(id=side + '_destroyed',
                     op='filter',
                     source=side,
                     concept='destroyed',
                     negate=False),
                dict(id=side + '_fraction', op='fraction', source=side + '_destroyed', total=side),
            ])
        steps.append(
            dict(id='answer', op='compare', source='left_fraction', target='right_fraction'))
    elif kind in ('nearest', 'distance'):
        steps.extend([
            dict(id='roads', op='select', kind='roads'),
            dict(id='eligible_roads',
                 op='filter',
                 source='roads',
                 concept='blocked' if kind == 'nearest' else 'clear',
                 negate=False),
        ])
        if kind == 'nearest':
            steps.extend([
                dict(id='closest',
                     op='nearest',
                     source='buildings',
                     target='eligible_roads',
                     policy='footprint_or_centerline'),
                dict(id='answer', op='ids', source='closest'),
            ])
        else:
            if not entity_id:
                raise ValueError('Supply the highlighted building ID')
            steps.extend([
                dict(id='highlighted', op='entity', source='buildings', entity_id=entity_id),
                dict(id='answer',
                     op='distance',
                     source='highlighted',
                     target='eligible_roads',
                     policy='footprint_or_centerline'),
            ])
    else:
        raise ValueError('Unknown paper example: ' + kind)
    return dict(format=FORMAT, steps=steps, output='answer')
