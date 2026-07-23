"""MD helpers — **out of scope for this repo's design campaign**.

The v0.6 funnel ends at AF + mechanistic ranking. Nothing in the CLI or
``run_design_pipeline`` proceeds to MD. This module is retained only as a
stub if an external follow-up ever imports metrics; do not call it from
the case-study path.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from catalyst_atlas.paths import PROCESSED, REPORTS, ensure_dirs

logger = logging.getLogger(__name__)


def write_md_plan(
    pairs: list[dict[str, str]],
    *,
    ns: float = 50.0,
) -> Path:
    """Document which systems to simulate — does not launch OpenMM."""
    ensure_dirs()
    plan = {
        "ns_per_system": ns,
        "pairs": pairs,
        "metrics": [
            "catalytic_pairwise_distances",
            "pocket_rmsd_vs_wt",
            "hbond_occupancy_shell",
            "ligand_or_metal_contact_persistence",
        ],
        "note": (
            "Optional high-value deep dive after AF funnel. "
            "Run externally with OpenMM; import results via import_md_metrics."
        ),
    }
    path = PROCESSED / "design" / "md_plan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2))
    logger.info("Wrote MD plan (%d pairs) → %s", len(pairs), path)
    return path


def import_md_metrics(path: Path) -> Path:
    """Copy/normalize an external MD metrics JSON into out/."""
    ensure_dirs()
    payload: dict[str, Any] = json.loads(Path(path).read_text())
    out = REPORTS / "design_md_deepdive.json"
    out.write_text(json.dumps(payload, indent=2))
    logger.info("Imported MD metrics → %s", out)
    return out
