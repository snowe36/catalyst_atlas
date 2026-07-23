"""Curated redesign panel — mechanism-aware diversity, not a metalloprotease pile-up."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from catalyst_atlas.design.pocket import build_pocket
from catalyst_atlas.paths import PROCESSED, RAW

logger = logging.getLogger(__name__)

# Preferred public M-CSA anchors + role tags for the case-study story.
PREFERRED_PANEL: list[dict[str, str]] = [
    {
        "enzyme_id": "MCSA00176",
        "role": "metalloprotease",
        "note": "thermolysin — Zn hydrolysis hero",
    },
    {
        "enzyme_id": "MCSA00623",
        "role": "metalloprotease",
        "note": "neprilysin — convergent Zn chemistry",
    },
    {
        "enzyme_id": "MCSA00661",
        "role": "cofactor_dependent",
        "note": "arylsulfatase — Ca/Mg",
    },
    {
        "enzyme_id": "MCSA00034",
        "role": "redox",
        "note": "catechol 2,3-dioxygenase — Fe redox",
    },
]

# Role quotas for auto-fill when preferred IDs are missing (demo atlas).
ROLE_TARGETS: dict[str, int] = {
    "metalloprotease": 2,
    "redox": 2,
    "transferase": 2,
    "cofactor_dependent": 2,
    "other": 2,
}

MIN_CATALYTIC = 3
MIN_REDESIGNABLE = 8
MAX_REDESIGNABLE = 40


def _role_for_row(row: pd.Series) -> str:
    fam = str(row.get("chemistry_family") or row.get("chemistry_class") or "").lower()
    mech = str(row.get("mechanistic_pattern") or "").lower()
    cof = str(row.get("cofactor_tags") or "").lower()
    if "transfer" in fam or fam.startswith("ec2") or "transferase" in fam:
        return "transferase"
    if "oxidation" in fam or "redox" in fam or "reduction" in fam:
        return "redox"
    if "hydrolysis" in fam and ("zn" in cof or "metal" in mech):
        return "metalloprotease"
    if cof and cof != "none":
        return "cofactor_dependent"
    return "other"


def passes_pocket_qc(pocket: dict[str, Any]) -> bool:
    n_cat = int(pocket.get("n_catalytic") or 0)
    n_red = int(pocket.get("n_redesignable") or 0)
    if n_cat < MIN_CATALYTIC:
        return False
    if n_red < MIN_REDESIGNABLE or n_red > MAX_REDESIGNABLE:
        return False
    # Need seq_index on catalytic residues for fixed-position invariants.
    for r in pocket.get("catalytic_residues") or []:
        if r.get("seq_index") is None:
            return False
    mapped = sum(1 for r in (pocket.get("redesignable") or []) if r.get("seq_index") is not None)
    return mapped >= MIN_REDESIGNABLE


def _load_atlas() -> pd.DataFrame:
    path = RAW / "catalytic_atlas.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run cat-download first")
    return pd.read_parquet(path)


DEFAULT_PANEL = [e["enzyme_id"] for e in PREFERRED_PANEL]


def resolve_panel(
    atlas: pd.DataFrame | None = None,
    *,
    target_size: int = 10,
    preferred: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Resolve a diverse panel from preferred IDs, filling gaps from the atlas."""
    atlas = atlas if atlas is not None else _load_atlas()
    preferred = preferred if preferred is not None else PREFERRED_PANEL
    by_id = {str(r["enzyme_id"]): r for _, r in atlas.iterrows()}

    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    role_counts: dict[str, int] = {k: 0 for k in ROLE_TARGETS}

    for pref in preferred:
        eid = pref["enzyme_id"]
        if eid not in by_id:
            continue
        row = by_id[eid]
        pocket = build_pocket(row)
        if not passes_pocket_qc(pocket):
            logger.warning("Preferred %s failed pocket QC; skipping", eid)
            continue
        role = pref.get("role") or _role_for_row(row)
        selected.append(
            {
                "enzyme_id": eid,
                "role": role,
                "note": pref.get("note", ""),
                "chemistry_family": pocket["reaction"]["chemistry_family"],
                "mechanistic_pattern": pocket["reaction"]["mechanistic_pattern"],
                "n_catalytic": pocket["n_catalytic"],
                "n_redesignable": pocket["n_redesignable"],
                "pdb_id": pocket.get("pdb_id", ""),
            }
        )
        used.add(eid)
        role_counts[role] = role_counts.get(role, 0) + 1

    # Auto-fill by role quotas for chemistry diversity.
    candidates: list[tuple[str, str, dict[str, Any], pd.Series]] = []
    for _, row in atlas.iterrows():
        eid = str(row["enzyme_id"])
        if eid in used:
            continue
        pocket = build_pocket(row)
        if not passes_pocket_qc(pocket):
            continue
        role = _role_for_row(row)
        candidates.append((eid, role, pocket, row))

    # Prefer filling under-quota roles first.
    for role, quota in ROLE_TARGETS.items():
        while role_counts.get(role, 0) < quota and len(selected) < target_size:
            pick = next((c for c in candidates if c[1] == role and c[0] not in used), None)
            if pick is None:
                break
            eid, role, pocket, _row = pick
            selected.append(
                {
                    "enzyme_id": eid,
                    "role": role,
                    "note": "auto-filled for chemistry diversity",
                    "chemistry_family": pocket["reaction"]["chemistry_family"],
                    "mechanistic_pattern": pocket["reaction"]["mechanistic_pattern"],
                    "n_catalytic": pocket["n_catalytic"],
                    "n_redesignable": pocket["n_redesignable"],
                    "pdb_id": pocket.get("pdb_id", ""),
                }
            )
            used.add(eid)
            role_counts[role] = role_counts.get(role, 0) + 1

    # Top up to target_size with any remaining QC-pass enzymes.
    for eid, role, pocket, _row in candidates:
        if len(selected) >= target_size:
            break
        if eid in used:
            continue
        selected.append(
            {
                "enzyme_id": eid,
                "role": role,
                "note": "auto-filled",
                "chemistry_family": pocket["reaction"]["chemistry_family"],
                "mechanistic_pattern": pocket["reaction"]["mechanistic_pattern"],
                "n_catalytic": pocket["n_catalytic"],
                "n_redesignable": pocket["n_redesignable"],
                "pdb_id": pocket.get("pdb_id", ""),
            }
        )
        used.add(eid)

    if not selected:
        raise RuntimeError("No enzymes passed pocket QC for the redesign panel")

    out_path = PROCESSED / "design" / "panel.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selected).to_json(out_path, orient="records", indent=2)
    logger.info(
        "Design panel n=%d roles=%s → %s",
        len(selected),
        dict(role_counts),
        out_path,
    )
    return selected
