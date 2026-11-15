"""Validate and execute ordered program steps on entity records.
Each step calls a supported operation and can read the results of earlier steps."""
from __future__ import annotations
import math
import re
import operator
from siren.concepts.registry import CONCEPT_BY_NAME
from siren.settings import BLOCKED_COUNT_THRESHOLD, LARGE_AREA_PX, QUERY_THRESHOLDS
from siren.programs.geometry import near, shape

FORMAT = 'siren.program'
# Geometry policies (programs/geometry.py): footprint uses the first footprint ring;
# footprint_or_centerline uses all valid rings, else the road centerline, else the box.
POLICIES = ('footprint', 'footprint_or_centerline')
# Each operation maps to (fields required besides id and op,
# {reference field: type of the earlier step it names}, type of the result).
OPERATIONS = {
    'select': ({'kind'}, {}, 'entities'),
    'filter': ({'source', 'concept', 'negate'}, {
        'source': 'entities'
    }, 'entities'),
    'relate': ({'source', 'target', 'distance_px', 'policy', 'mode'}, {
        'source': 'entities',
        'target': 'entities'
    }, 'entities'),
    'extreme': ({'source', 'selection'}, {
        'source': 'entities'
    }, 'entities'),
    'count': ({'source'}, {
        'source': 'entities'
    }, 'number'),
    'exists': ({'source'}, {
        'source': 'entities'
    }, 'boolean'),
    'ids': ({'source'}, {
        'source': 'entities'
    }, 'ids'),
    'fraction': ({'source', 'total'}, {
        'source': 'entities',
        'total': 'entities'
    }, 'number'),
    'count_density': ({'source'}, {
        'source': 'entities'
    }, 'number'),
    'half': ({'source', 'side'}, {
        'source': 'entities'
    }, 'entities'),
    'entity': ({'source', 'entity_id'}, {
        'source': 'entities'
    }, 'entities'),
    'nearest': ({'source', 'target', 'policy'}, {
        'source': 'entities',
        'target': 'entities'
    }, 'entities'),
    'distance': ({'source', 'target', 'policy'}, {
        'source': 'entities',
        'target': 'entities'
    }, 'number'),
    'threshold': ({'source', 'value', 'comparison'}, {
        'source': 'number'
    }, 'boolean'),
    'compare': ({'source', 'target'}, {
        'source': 'number',
        'target': 'number'
    }, 'comparison')
}
# Units of the numeric operations; other results have no unit.
UNITS = {'count': 'entities', 'fraction': 'fraction', 'count_density': 'entities/km²', 'distance': 'm'}
COMPARISONS = {
    'ge': operator.ge,
    'gt': operator.gt,
    'le': operator.le,
    'lt': operator.lt,
    'eq': operator.eq
}
# Concepts a filter may use for each entity kind. large, intact and clear are derived from
# geometry or other scores; blocked_for_counting is blocked at the counting threshold.
FILTER_CONCEPTS = {
    'buildings': ['damaged', 'destroyed', 'residential', 'surrounded_by_vegetation', 'large', 'intact'],
    'roads': ['blocked', 'clear', 'blocked_for_counting']
}
VISUAL_CONCEPTS = {
    'damaged': 'is_damaged',
    'destroyed': 'is_destroyed',
    'residential': 'is_residential',
    'surrounded_by_vegetation': 'surrounded_by_vegetation',
    'blocked': 'is_road_blocked'
}
# Concept -> (exported score attribute, threshold); the concept holds when score >= threshold.
SCORES = {
    name: (CONCEPT_BY_NAME[concept].score_key, QUERY_THRESHOLDS[concept])
    for (name, concept) in VISUAL_CONCEPTS.items()
}
SCORES['blocked_for_counting'] = (CONCEPT_BY_NAME['is_road_blocked'].score_key, BLOCKED_COUNT_THRESHOLD)


def finite(value, minimum=0, name='Value'):
    """Return value if it is a finite number >= minimum (Booleans rejected); otherwise raise."""
    if (isinstance(value, bool) or not isinstance(value, (float, int))
            or not math.isfinite(value) or value < minimum):
        raise ValueError(f'{name} must be a finite number >= {minimum}, got {value!r}')
    return value


def _context(context, key):
    """A required positive execution-context value, such as pixel_size_m."""
    return finite(context.get(key), minimum=1e-12, name='Execution context ' + key)


def validate(program):
    """Check a program's envelope, step fields, references, types and units without running it."""
    if (not isinstance(program, dict) or set(program) != {'format', 'steps', 'output'}
            or program['format'] != FORMAT):
        raise ValueError('Invalid program envelope')
    steps = program['steps']
    if not isinstance(steps, list) or not 1 <= len(steps) <= 64:
        raise ValueError('Expected 1–64 steps')
    # Result type, entity kind and operation of each earlier step, by step ID.
    (types, kinds, ops) = ({}, {}, {})
    halves = set()  # Steps whose entities are limited to one image half.
    for step in steps:
        if (not isinstance(step, dict) or not isinstance(step.get('op'), str)
                or step['op'] not in OPERATIONS):
            raise ValueError('Unknown operation')
        (fields, refs, output_type) = OPERATIONS[step['op']]
        if set(step) != fields | {'id', 'op'}:
            raise ValueError('Unexpected or missing operation fields')
        for field in ('kind', 'concept', 'policy', 'mode', 'side', 'comparison', 'selection'):
            if field in step and not isinstance(step[field], str):
                raise ValueError(f'{field} must be a string')
        name = step['id']
        if (not isinstance(name, str) or not re.fullmatch('[a-z][a-z0-9_]{0,39}', name)
                or name in types):
            raise ValueError('Invalid/duplicate step ID')
        for (field, expected) in refs.items():
            ref = step[field]
            if not isinstance(ref, str) or types.get(ref) != expected:
                raise ValueError('Reference must be an earlier step of type ' + expected)
        op = step['op']
        kind = step.get('kind') if op == 'select' else kinds.get(step.get('source'))
        if op == 'select' and kind not in FILTER_CONCEPTS:
            raise ValueError('Unsupported entity kind')
        if op == 'filter' and (step['concept'] not in FILTER_CONCEPTS[kind]
                               or type(step['negate']) is not bool):
            raise ValueError('Unsupported concept or invalid negation')
        # The policy picks the geometry. footprint uses building footprints, so both sets
        # must be buildings.
        if op in ('relate', 'nearest', 'distance'):
            if step['policy'] not in POLICIES:
                raise ValueError('Unknown geometry policy')
            if step['policy'] == 'footprint' and (
                    kind != 'buildings' or kinds[step['target']] != 'buildings'):
                raise ValueError('The footprint policy requires two building sets')
        if op == 'relate':
            finite(step['distance_px'], name='distance_px')
            if step['mode'] not in ('any', 'none'):
                raise ValueError('Unknown relation mode')
        if op == 'half' and step['side'] not in ('left', 'right'):
            raise ValueError('Choose the left or right image half')
        if op == 'entity' and (not isinstance(step['entity_id'], str)
                               or not step['entity_id'].strip()):
            raise ValueError('Supply an entity ID')
        if op == 'threshold':
            finite(step['value'], name='Threshold value')
            if step['comparison'] not in COMPARISONS:
                raise ValueError('Unknown numeric comparison')
        if op == 'extreme' and step['selection'] not in ('largest', 'smallest'):
            raise ValueError('Unknown area selection')
        if op == 'extreme' and kind != 'buildings':
            raise ValueError('extreme compares building footprint areas; roads have none')
        if op == 'fraction' and kind != kinds[step['total']]:
            raise ValueError('Fraction requires matching entity kinds')
        # Density divides by the whole valid image area.
        if op == 'count_density' and step['source'] in halves:
            raise ValueError('count_density covers the whole image only; it cannot follow half')
        if op == 'compare' and UNITS[ops[step['source']]] != UNITS[ops[step['target']]]:
            raise ValueError('Compared quantities must have matching units')
        if op == 'half' or step.get('source') in halves:
            halves.add(name)
        (types[name], kinds[name], ops[name]) = (output_type, kind, op)
    output = program['output']
    if not isinstance(output, str) or output not in types or types[output] == 'entities':
        raise ValueError('Output must name a step that returns a number, Boolean, ID list or comparison')
    return program


def required_steps(program):
    """Return the steps that the output depends on, in program order."""
    by_id = {step['id']: step for step in program['steps']}
    (needed, pending) = (set(), [program['output']])
    while pending:
        name = pending.pop()
        if name in needed:
            continue
        needed.add(name)
        step = by_id[name]
        pending.extend((step[field] for field in OPERATIONS[step['op']][1]))
    return [step for step in program['steps'] if step['id'] in needed]


def _area(row):
    """Footprint area: the exported mask area, geometry.area_px."""
    geometry = row.get('geometry')
    area = geometry.get('area_px') if isinstance(geometry, dict) else None
    return finite(area, name=f"geometry.area_px of {row.get('id')}")


def _predicate(row, concept):
    """Whether a filter concept holds for one entity."""
    if concept == 'large':
        return _area(row) >= LARGE_AREA_PX
    if concept == 'intact':
        return not _predicate(row, 'damaged') and not _predicate(row, 'destroyed')
    if concept == 'clear':
        return not _predicate(row, 'blocked')
    # Read a probability, not a stored yes/no label. Missing scores are errors.
    (field, threshold) = SCORES[concept]
    value = finite((row.get('attributes') or {}).get(field), name=f"{field} of {row.get('id')}")
    if value > 1:
        raise ValueError(f"{field} of {row.get('id')} is outside [0,1]")
    return value >= threshold


def nearest_distances(rows, targets, policy):
    """Return each row's shortest distance in pixels to a target other than itself.

    None marks a row with no other target."""
    if not targets:
        return [None] * len(rows)
    target_shapes = [(target, shape(target, policy)) for target in targets]
    distances = []
    for row in rows:
        geometry = shape(row, policy)
        # Steps pass the scene's own row objects, so identity marks the same entity.
        found = [geometry.distance(other) for (target, other) in target_shapes if target is not row]
        distances.append(min(found) if found else None)
    return distances


def _execute_step(step, groups, values, context):
    """Apply one validated operation and return its value."""
    op = step['op']
    source = values.get(step.get('source'))
    if op == 'select':
        value = groups.get(step['kind'])
        if not isinstance(value, list):
            raise ValueError('Scene has no ' + step['kind'] + ' list')
        if not all(isinstance(r, dict) for r in value):
            raise ValueError('Malformed entity in ' + step['kind'] + ': expected an object')
        ids = [r.get('id') for r in value]
        if any(not isinstance(i, str) or not i.strip() for i in ids) or len(ids) != len(set(ids)):
            raise ValueError('Missing/duplicate entity ID in ' + step['kind'])
    elif op == 'filter':
        value = [r for r in source if _predicate(r, step['concept']) != step['negate']]
    elif op == 'half':
        width = _context(context, 'image_width_px')
        value = []
        for row in source:
            box = (row.get('geometry') or {}).get('bbox_xyxy')
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                raise ValueError('Image-half selection requires bounding boxes')
            (left, _, right, _) = [finite(v, name='bbox_xyxy value') for v in box]
            if right < left:
                raise ValueError('Invalid bounding box')
            # Box center left of the midpoint: left half; the midpoint itself is in the right half.
            if ((left + right) / 2 < width / 2) == (step['side'] == 'left'):
                value.append(row)
    elif op == 'entity':
        value = [row for row in source if row['id'] == step['entity_id']]
        if not value:
            raise ValueError('Requested entity is absent: ' + step['entity_id'])
    elif op == 'relate':
        targets = values[step['target']]
        # An entity is never its own target (rows are the scene's objects in every step).
        want_near = step['mode'] == 'any'
        value = [
            r for r in source
            if any(t is not r and near(r, t, step['distance_px'], step['policy'])
                   for t in targets) == want_near
        ]
    elif op == 'extreme':
        areas = [_area(r) for r in source]
        if any(a <= 0 for a in areas):
            raise ValueError('Selection requires positive footprint areas')
        best = (max if step['selection'] == 'largest' else min)(areas) if areas else None
        # Keep all entities tied for the selected area.
        value = [r for (r, a) in zip(source, areas) if a == best]
    elif op in ('nearest', 'distance'):
        distances = nearest_distances(source, values[step['target']], step['policy'])
        found = [d for d in distances if d is not None]
        best = min(found) if found else None
        if op == 'nearest':
            value = [row for (row, distance) in zip(source, distances)
                     if distance is not None and distance == best]
        else:
            pixel_size = _context(context, 'pixel_size_m')
            value = best * pixel_size if best is not None else None  # pixels to meters
    elif op == 'count':
        value = len(source)
    elif op == 'exists':
        value = bool(source)
    elif op == 'ids':
        value = [r['id'] for r in source]
    elif op == 'fraction':
        total = values[step['total']]
        if not {r['id'] for r in source} <= {r['id'] for r in total}:
            raise ValueError('Numerator must be a subset of denominator')
        # An empty denominator gives an undefined answer.
        value = len(source) / len(total) if total else None
    elif op == 'count_density':
        area = _context(context, 'valid_area_px')
        pixel_size = _context(context, 'pixel_size_m')
        # Entities per km²: the valid image area converted from pixels to square kilometers.
        value = len(source) / (area * pixel_size * pixel_size / 1e6)
    elif op == 'threshold':
        compare = COMPARISONS[step['comparison']]
        value = compare(source, step['value']) if source is not None else None
    elif op == 'compare':
        target = values[step['target']]
        if source is None or target is None:
            value = None
        elif source > target:
            value = 'greater'
        elif source < target:
            value = 'less'
        else:
            value = 'equal'
    return value


def execute(program, scene, *, context=None):
    """Validate and run a program on one scene; return the answer, its unit and the trace."""
    # Check every step before skipping branches the answer does not use.
    validate(program)
    steps = required_steps(program)
    if not isinstance(scene, dict):
        raise ValueError('Expected scene object')
    groups = scene.get('entities', {})
    if not isinstance(groups, dict):
        raise ValueError('Expected an entities object')
    context = context or {}
    (values, trace, units) = ({}, [], {})
    for step in steps:
        (op, name) = (step['op'], step['id'])
        value = _execute_step(step, groups, values, context)
        (values[name], units[name]) = (value, UNITS.get(op))
        # Store entity IDs in the trace instead of repeating full records.
        trace.append({
            'id': name,
            'op': op,
            'value': [r['id'] for r in value]
            if isinstance(value, list) and value and isinstance(value[0], dict) else value,
            'unit': units[name]
        })
    answer = values[program['output']]
    return {
        'status': 'undefined' if answer is None else 'ok',
        'value': answer,
        'reason': 'No eligible entities or an empty denominator' if answer is None else '',
        'unit': units[program['output']],
        'trace': trace,
        'unused_steps': [step['id'] for step in program['steps'] if step not in steps]
    }
