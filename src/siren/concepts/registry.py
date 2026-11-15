"""The five visual classifiers used by SIREN: entity kind, exported score key and input images.

Damage and destruction are scored by the convolutional heads in siren.inference.buildings;
their entries here only record the score key, entity kind and paired input."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

InputMode = Literal["paired", "single"]


@dataclass(frozen=True)
class ConceptSpec:
    """One classifier: its entity kind (which box file its labels use), exported score key and
    input images."""
    name: str
    entity_group: str
    score_key: str
    input_mode: InputMode
    classes: tuple[str, str]

    @property
    def head_input_dim(self) -> int:
        """MLP head input size: 192 (pre, post and absolute difference, 64 each) if paired, else 64."""
        return 192 if self.input_mode == "paired" else 64

    @property
    def is_paired(self) -> bool:
        """Return whether this classifier uses both image dates."""
        return self.input_mode == "paired"


CONCEPTS: list[ConceptSpec] = [
    ConceptSpec(
        name="is_damaged",
        entity_group="buildings",
        score_key="is_damaged_score",
        input_mode="paired",
        classes=("intact", "damaged"),
    ),
    ConceptSpec(
        name="is_destroyed",
        entity_group="buildings",
        score_key="is_destroyed_score",
        input_mode="paired",
        classes=("not_destroyed", "destroyed"),
    ),
    ConceptSpec(
        name="is_residential",
        entity_group="buildings",
        score_key="is_residential_score",
        input_mode="single",
        classes=("non_residential", "residential"),
    ),
    ConceptSpec(
        name="is_road_blocked",
        entity_group="roads",
        score_key="is_road_blocked_score",
        input_mode="paired",
        classes=("clear", "blocked"),
    ),
    ConceptSpec(
        name="surrounded_by_vegetation",
        entity_group="buildings",
        score_key="surrounded_by_vegetation_score",
        input_mode="single",
        classes=("not_surrounded", "surrounded"),
    ),
]


CONCEPT_BY_NAME: dict[str, ConceptSpec] = {c.name: c for c in CONCEPTS}


def get_concepts(names: list[str] | None = None) -> list[ConceptSpec]:
    """Return the specs of the given names, in order, or every spec when names is None."""
    if names is None:
        return list(CONCEPTS)
    out = []
    for n in names:
        if n not in CONCEPT_BY_NAME:
            raise KeyError(f"Unknown concept {n!r}; choose from {list(CONCEPT_BY_NAME)}")
        out.append(CONCEPT_BY_NAME[n])
    return out

