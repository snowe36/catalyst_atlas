from __future__ import annotations

from functools import lru_cache
from typing import Any

import yaml

from catalyst_atlas.paths import RESOURCES


@lru_cache(maxsize=1)
def load_ontology() -> dict[str, Any]:
    path = RESOURCES / "chemistry_ontology.yaml"
    with path.open() as fh:
        return yaml.safe_load(fh)


def chemistry_classes() -> list[str]:
    return list(load_ontology()["chemistry_classes"])


def chemistry_families() -> list[str]:
    data = load_ontology()
    if "chemistry_families" in data:
        return list(data["chemistry_families"])
    return []


def mechanistic_patterns() -> list[str]:
    data = load_ontology()
    if "mechanistic_patterns" in data:
        return list(data["mechanistic_patterns"])
    return []


def families() -> list[dict[str, Any]]:
    return list(load_ontology()["families"])
