"""Optional MD deep-dive for top WT/design pairs (OpenMM).

Not required for the AF funnel case study. Use for 1–2 enzymes after
mechanistic ranking to compare catalytic distances / pocket RMSD.

Expected workflow (external GPU):
  1. Pick WT + top design PDB from AF outputs
  2. Minimize + 50 ns production (or shorter smoke)
  3. Write md_metrics.json consumed by the design report
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
    """Copy/normalize an external MD metrics JSON into reports/."""
    ensure_dirs()
    payload: dict[str, Any] = json.loads(Path(path).read_text())
    out = REPORTS / "design_md_deepdive.json"
    out.write_text(json.dumps(payload, indent=2))
    logger.info("Imported MD metrics → %s", out)
    return out
