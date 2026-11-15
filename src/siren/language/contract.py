"""Operation descriptions and API response schema for the operations defined in programs/engine.py."""
from siren.programs.engine import FILTER_CONCEPTS, FORMAT, OPERATIONS, POLICIES

DESCRIPTIONS = {
    'select': 'Return all entities of the given kind.',
    'filter': 'Keep source entities that have the concept (negate=false) or lack it (negate=true). Chain filters for AND.',
    'relate': 'Keep source entities with at least one target within distance_px pixels (mode=any) or with none (mode=none). An entity is never its own target. Use only for spatial conditions.',
    'extreme': 'Return the source buildings with the largest or smallest footprint area, keeping ties.',
    'count': 'Return the number of source entities.',
    'exists': 'Return whether the source set is nonempty.',
    'ids': 'Return the IDs of the source entities.',
    'fraction': 'Return count(source) / count(total); source must be a subset of total.',
    'count_density': 'Return count(source) per square kilometer of the image; whole image only, so never after half.',
    'half': 'Keep source entities whose bounding-box center is in the left or right half of the image; the midpoint belongs to the right half.',
    'entity': 'Select the source entity whose ID the question gives. If no ID is given, return program=null and ask for it in reason.',
    'nearest': 'Return the source entities closest to any other target entity, keeping ties; empty if there is no such pair.',
    'distance': 'Return the shortest distance in meters between a source entity and a different target entity; undefined if there is no such pair.',
    'threshold': 'Compare a number with value, in the same units, using ge, gt, le, lt or eq. Use only a value stated in the question.',
    'compare': 'Compare two numbers with the same units; return greater, less or equal for source relative to target.',
}


def response_schema():
    """JSON schema of a compiler response: a reason and a program (or null)."""
    text = {'type': 'string'}
    fields = {
        'kind': {
            'type': 'string',
            'enum': list(FILTER_CONCEPTS)
        },
        'concept': {
            'type': 'string',
            'enum': sorted({c
                            for cs in FILTER_CONCEPTS.values()
                            for c in cs})
        },
        'negate': {
            'type': 'boolean',
            'description': 'True means exclude entities satisfying the concept; false means keep them.'
        },
        'distance_px': {
            'type': 'number'
        },
        'policy': {
            'type': 'string',
            'enum': list(POLICIES)
        },
        'mode': {
            'type': 'string',
            'enum': ['any', 'none']
        },
        'selection': {
            'type': 'string',
            'enum': ['largest', 'smallest']
        },
        'side': {
            'type': 'string',
            'enum': ['left', 'right']
        },
        'entity_id': text,
        'value': {
            'type': 'number'
        },
        'comparison': {
            'type': 'string',
            'enum': ['ge', 'gt', 'le', 'lt', 'eq']
        },
    }
    # One step schema per operation in the executor's registry.
    variants = []
    for op, (required, refs, _) in OPERATIONS.items():
        props = {'id': text, 'op': {'type': 'string', 'enum': [op]}}
        props.update({key: fields.get(key, text) for key in sorted(required)})
        variants.append({
            'type': 'object',
            'properties': props,
            'required': list(props),
            'additionalProperties': False,
        })
    program = {
        'type': 'object',
        'properties': {
            'format': {
                'type': 'string',
                'enum': [FORMAT]
            },
            'steps': {
                'type': 'array',
                'items': {
                    'anyOf': variants
                }
            },
            'output': text,
        },
        'required': ['format', 'steps', 'output'],
        'additionalProperties': False,
    }
    # A null program carries the reason a question needs clarification or support.
    return {
        'type': 'object',
        'properties': {
            'reason': text,
            'program': {
                'anyOf': [program, {
                    'type': 'null'
                }]
            }
        },
        'required': ['reason', 'program'],
        'additionalProperties': False,
    }
