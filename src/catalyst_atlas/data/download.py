"""Download / materialize the high-confidence catalytic atlas.

- ``demo=True``: synthetic atlas for CI / offline work
- ``demo=False``: public M-CSA curated sites + RCSB structures
"""

from __future__ import annotations

import logging
from pathlib import Path

from catalyst_atlas.data.generate_demo import generate_demo_atlas, save_raw_atlas
from catalyst_atlas.paths import ensure_dirs

logger = logging.getLogger(__name__)


def download_atlas(
    demo: bool = True,
    n_enzymes: int = 800,
    seed: int = 7,
) -> Path:
    """Materialize the catalytic atlas under ``data/raw/``.

    Parameters
    ----------
    demo:
        If True, generate the synthetic high-confidence demo atlas.
        If False, ingest curated M-CSA entries with real PDB coordinates.
    n_enzymes:
        Cap on number of enzymes (demo size, or first N M-CSA ids).
    """
    ensure_dirs()
    if demo:
        df = generate_demo_atlas(n_enzymes=n_enzymes, seed=seed)
    else:
        from catalyst_atlas.data.mcsa import build_mcsa_atlas

        logger.info(
            "Ingesting public M-CSA catalytic sites (n_enzymes=%s) — "
            "this downloads API JSON, UniProt sequences, and RCSB PDBs",
            n_enzymes,
        )
        try:
            df = build_mcsa_atlas(n_enzymes=n_enzymes, seed=seed)
        except Exception:
            logger.exception(
                "M-CSA ingest failed; falling back to synthetic demo atlas"
            )
            df = generate_demo_atlas(n_enzymes=n_enzymes, seed=seed)

    path = save_raw_atlas(df)
    logger.info("Wrote %s (%d enzymes, source=%s)", path, len(df), df["source"].iloc[0])
    return path
