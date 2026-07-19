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
    expanded: bool = False,
    n_extra: int = 200,
    allow_alphafold: bool = True,
) -> Path:
    """Materialize the catalytic atlas under ``data/raw/``.

    Parameters
    ----------
    demo:
        If True, generate the synthetic high-confidence demo atlas.
        If False, ingest curated M-CSA entries with real PDB coordinates.
    n_enzymes:
        Cap on number of enzymes (demo size, or first N M-CSA ids).
    expanded:
        If True (public mode), merge UniProt ACT_SITE extras + EC labels /
        ``structure_source`` stratification (experimental vs AlphaFold).
    n_extra:
        Max UniProt-sourced extras when ``expanded=True``.
    allow_alphafold:
        When expanding, allow AFDB models for UniProt IDs lacking PDB.
    """
    ensure_dirs()
    if demo:
        df = generate_demo_atlas(n_enzymes=n_enzymes, seed=seed)
        from catalyst_atlas.data.uniprot_expand import attach_ec_labels

        df = attach_ec_labels(df)
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

        if expanded:
            from catalyst_atlas.data.uniprot_expand import merge_expanded_atlas

            logger.info(
                "Expanding beyond M-CSA with UniProt ACT_SITE sites "
                "(n_extra=%s, allow_alphafold=%s)",
                n_extra,
                allow_alphafold,
            )
            try:
                df = merge_expanded_atlas(
                    df, n_extra=n_extra, seed=seed, allow_alphafold=allow_alphafold
                )
            except Exception:
                logger.exception("UniProt expand failed; keeping M-CSA base with EC labels")
                from catalyst_atlas.data.uniprot_expand import attach_ec_labels

                df = attach_ec_labels(df)
        else:
            from catalyst_atlas.data.uniprot_expand import attach_ec_labels

            df = attach_ec_labels(df)

    path = save_raw_atlas(df)
    src = df["source"].value_counts().to_dict() if "source" in df.columns else {}
    logger.info("Wrote %s (%d enzymes, sources=%s)", path, len(df), src)
    return path
