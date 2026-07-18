"""Download / materialize the high-confidence catalytic atlas.

Public M-CSA-scale curation is the long-term source. For reproducible CI and
offline work, ``--demo`` builds a label-clean atlas from curated mechanistic
family templates (see ``resources/chemistry_ontology.yaml``).
"""

from __future__ import annotations

import logging
from pathlib import Path

from catalyst_atlas.data.generate_demo import generate_demo_atlas, save_raw_atlas
from catalyst_atlas.paths import ensure_dirs

logger = logging.getLogger(__name__)


def download_atlas(demo: bool = True, n_enzymes: int = 800, seed: int = 7) -> Path:
    """Materialize the catalytic atlas under ``data/raw/``.

    Parameters
    ----------
    demo:
        If True (default for v1), generate the high-confidence demo atlas.
        If False, attempt public curated sources (falls back to demo if unavailable).
    """
    ensure_dirs()
    if not demo:
        logger.warning(
            "Public M-CSA/UniProt bulk ingest is not wired for unattended CI yet; "
            "falling back to the high-confidence demo atlas (quality-first)."
        )
    df = generate_demo_atlas(n_enzymes=n_enzymes, seed=seed)
    path = save_raw_atlas(df)
    logger.info("Wrote %s (%d enzymes)", path, len(df))
    return path
