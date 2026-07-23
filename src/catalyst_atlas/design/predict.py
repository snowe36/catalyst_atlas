"""Structure prediction adapter (AF2 / ColabFold) — external, importable metrics."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from catalyst_atlas.paths import PROCESSED, ensure_dirs

logger = logging.getLogger(__name__)


def prediction_dir(enzyme_id: str, design_id: str) -> Path:
    return PROCESSED / "design" / "predictions" / enzyme_id / design_id


def write_prediction_metrics(
    enzyme_id: str,
    design_id: str,
    *,
    mean_plddt: float,
    pocket_pae: float | None = None,
    pdb_path: Path | None = None,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Persist AF/ColabFold-style metrics for one design (or WT)."""
    ensure_dirs()
    out = prediction_dir(enzyme_id, design_id)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "enzyme_id": enzyme_id,
        "design_id": design_id,
        "mean_plddt": float(mean_plddt),
        "pocket_pae": None if pocket_pae is None else float(pocket_pae),
        "pdb_path": str(pdb_path) if pdb_path else None,
        **(meta or {}),
    }
    path = out / "metrics.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_prediction_metrics(enzyme_id: str, design_id: str) -> dict[str, Any] | None:
    path = prediction_dir(enzyme_id, design_id) / "metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def import_metrics_table(path: Path) -> int:
    """Import a JSON list or JSONL of prediction metrics.

    Each record needs: enzyme_id, design_id, mean_plddt; optional pocket_pae.
    """
    text = path.read_text().strip()
    if not text:
        return 0
    if text.startswith("["):
        records = json.loads(text)
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    n = 0
    for rec in records:
        write_prediction_metrics(
            rec["enzyme_id"],
            rec["design_id"],
            mean_plddt=float(rec["mean_plddt"]),
            pocket_pae=rec.get("pocket_pae"),
            pdb_path=Path(rec["pdb_path"]) if rec.get("pdb_path") else None,
            meta={k: v for k, v in rec.items() if k not in {
                "enzyme_id", "design_id", "mean_plddt", "pocket_pae", "pdb_path"
            }},
        )
        n += 1
    logger.info("Imported %d prediction metric records from %s", n, path)
    return n


def structure_confidence_from_metrics(metrics: dict[str, Any] | None) -> float:
    """Map pLDDT / PAE into [0, 1] structure confidence (higher is better)."""
    if not metrics:
        # Neutral default when AF output is unavailable (sequence-only scoring).
        return 0.5
    plddt = float(metrics.get("mean_plddt") or 50.0)
    # pLDDT is typically 0–100.
    conf = max(0.0, min(1.0, plddt / 100.0))
    pae = metrics.get("pocket_pae")
    if pae is not None:
        # Lower PAE is better; map ~0–30 Å into a mild penalty.
        pae_term = max(0.0, min(1.0, 1.0 - float(pae) / 30.0))
        conf = 0.7 * conf + 0.3 * pae_term
    return float(conf)


def mock_prediction_metrics(
    enzyme_id: str,
    design_id: str,
    *,
    mean_plddt: float = 85.0,
    pocket_pae: float = 5.0,
) -> dict[str, Any]:
    """Write placeholder AF metrics for offline demos."""
    write_prediction_metrics(
        enzyme_id,
        design_id,
        mean_plddt=mean_plddt,
        pocket_pae=pocket_pae,
        meta={"source": "mock"},
    )
    return load_prediction_metrics(enzyme_id, design_id) or {}
