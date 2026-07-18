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


def families() -> list[dict[str, Any]]:
    return list(load_ontology()["families"])
