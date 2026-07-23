"""Phase 1 — catalytic pocket artifacts (source of truth for design indexing)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from catalyst_atlas.paths import PROCESSED, RAW, ensure_dirs
from catalyst_atlas.site.extract import FIRST_SHELL_RADIUS

logger = logging.getLogger(__name__)

SECOND_SHELL_RADIUS = 12.0
LIGAND_CONTACT_RADIUS = 6.0


def _parse_json(cell: Any) -> list[dict[str, Any]]:
    if isinstance(cell, list):
        return cell
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    if isinstance(cell, str):
        return json.loads(cell) if cell else []
    return []


def _residue_record(
    r: dict[str, Any],
    *,
    shell: str | None,
    seq_index: int | None,
    dist_to_core: float | None = None,
) -> dict[str, Any]:
    xyz = [float(x) for x in r["xyz"]]
    rec: dict[str, Any] = {
        "chain": str(r.get("chain") or "A"),
        "resnum": int(r["resnum"]),
        "aa": str(r["aa"]),
        "xyz": xyz,
    }
    if shell is not None:
        rec["shell"] = shell
    if seq_index is not None:
        rec["seq_index"] = int(seq_index)
    if dist_to_core is not None:
        rec["dist_to_core"] = float(dist_to_core)
    return rec


def _seq_index_for(resnum: int, sequence: str) -> int | None:
    if not sequence:
        return None
    idx = int(resnum) - 1
    if 0 <= idx < len(sequence):
        return idx
    return None


def build_pocket(row: pd.Series) -> dict[str, Any]:
    """Build a rich pocket artifact for one enzyme.

    Shell rules (CA distance to catalytic centroid):
    - first:  <= FIRST_SHELL_RADIUS (8 Å)
    - second: (8, SECOND_SHELL_RADIUS] Å
    Catalytic residues are never redesignable.
    """
    residues = _parse_json(row.get("site_residues_json"))
    ligands = _parse_json(row.get("ligands_json"))
    sequence = str(row.get("sequence") or "")

    catalytic_raw = [r for r in residues if r.get("role") == "catalytic"]
    if not catalytic_raw:
        catalytic_raw = list(residues)

    core = np.array([r["xyz"] for r in catalytic_raw], dtype=float)
    center = core.mean(axis=0)

    catalytic = [
        _residue_record(
            r,
            shell=None,
            seq_index=_seq_index_for(int(r["resnum"]), sequence),
        )
        for r in catalytic_raw
    ]
    catalytic_keys = {(c["chain"], c["resnum"]) for c in catalytic}

    redesignable: list[dict[str, Any]] = []
    for r in residues:
        key = (str(r.get("chain") or "A"), int(r["resnum"]))
        if key in catalytic_keys:
            continue
        d = float(np.linalg.norm(np.array(r["xyz"], dtype=float) - center))
        if d <= FIRST_SHELL_RADIUS:
            shell = "first"
        elif d <= SECOND_SHELL_RADIUS:
            shell = "second"
        else:
            continue
        redesignable.append(
            _residue_record(
                r,
                shell=shell,
                seq_index=_seq_index_for(int(r["resnum"]), sequence),
                dist_to_core=d,
            )
        )
    redesignable.sort(key=lambda x: (x.get("dist_to_core", 0.0), x["resnum"]))

    # Ligand contacts: cofactors, metals, and any other annotated ligands.
    contact_residues = catalytic + redesignable
    ligand_contacts: list[dict[str, Any]] = []
    for lig in ligands:
        lxyz = np.array(lig["xyz"], dtype=float)
        for r in contact_residues:
            d = float(np.linalg.norm(np.array(r["xyz"], dtype=float) - lxyz))
            if d <= LIGAND_CONTACT_RADIUS:
                ligand_contacts.append(
                    {
                        "ligand": lig.get("name"),
                        "ligand_kind": lig.get("kind"),
                        "chain": r["chain"],
                        "resnum": r["resnum"],
                        "aa": r["aa"],
                        "distance": d,
                    }
                )

    chem_family = row.get("chemistry_family") or row.get("chemistry_class") or "unknown"
    mech = row.get("mechanistic_pattern") or row.get("catalytic_pattern") or "unknown"

    return {
        "enzyme_id": str(row["enzyme_id"]),
        "pdb_id": str(row.get("pdb_id") or ""),
        "uniprot_id": str(row.get("uniprot_id") or ""),
        "enzyme_name": str(row.get("enzyme_name") or row.get("family_id") or ""),
        "sequence": sequence,
        "reaction": {
            "chemistry_family": str(chem_family),
            "mechanistic_pattern": str(mech),
            "ec_number": str(row.get("ec_number") or ""),
        },
        "core_centroid": center.tolist(),
        "catalytic_residues": catalytic,
        "redesignable": redesignable,
        "ligand_contacts": ligand_contacts,
        "ligands": ligands,
        "n_catalytic": len(catalytic),
        "n_redesignable": len(redesignable),
        "n_first_shell": sum(1 for r in redesignable if r.get("shell") == "first"),
        "n_second_shell": sum(1 for r in redesignable if r.get("shell") == "second"),
    }


def pocket_to_row(pocket: dict[str, Any]) -> dict[str, Any]:
    """Flatten pocket JSON for parquet storage."""
    return {
        "enzyme_id": pocket["enzyme_id"],
        "pdb_id": pocket.get("pdb_id", ""),
        "uniprot_id": pocket.get("uniprot_id", ""),
        "enzyme_name": pocket.get("enzyme_name", ""),
        "sequence": pocket.get("sequence", ""),
        "chemistry_family": pocket["reaction"]["chemistry_family"],
        "mechanistic_pattern": pocket["reaction"]["mechanistic_pattern"],
        "ec_number": pocket["reaction"].get("ec_number", ""),
        "n_catalytic": pocket["n_catalytic"],
        "n_redesignable": pocket["n_redesignable"],
        "n_first_shell": pocket["n_first_shell"],
        "n_second_shell": pocket["n_second_shell"],
        "pocket_json": json.dumps(pocket),
        "catalytic_residues_json": json.dumps(pocket["catalytic_residues"]),
        "redesignable_json": json.dumps(pocket["redesignable"]),
        "ligand_contacts_json": json.dumps(pocket["ligand_contacts"]),
    }


def load_pocket(enzyme_id: str, pockets_dir: Path | None = None) -> dict[str, Any]:
    path = (pockets_dir or (PROCESSED / "design" / "pockets")) / f"{enzyme_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing pocket artifact: {path}")
    return json.loads(path.read_text())


def run_pockets(
    raw_path: Path | None = None,
    enzyme_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Build pocket artifacts for the atlas (or a subset) and persist them."""
    ensure_dirs()
    path = raw_path or (RAW / "catalytic_atlas.parquet")
    if not path.exists():
        raise FileNotFoundError(f"Missing raw atlas at {path}; run cat-download first")

    atlas = pd.read_parquet(path)
    if enzyme_ids is not None:
        atlas = atlas[atlas["enzyme_id"].isin(enzyme_ids)].copy()
        if atlas.empty:
            raise ValueError(f"No atlas rows for enzyme_ids={enzyme_ids}")

    pockets_dir = PROCESSED / "design" / "pockets"
    pockets_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for _, row in atlas.iterrows():
        pocket = build_pocket(row)
        (pockets_dir / f"{pocket['enzyme_id']}.json").write_text(
            json.dumps(pocket, indent=2)
        )
        rows.append(pocket_to_row(pocket))

    out = pd.DataFrame(rows)
    out_path = PROCESSED / "design_pockets.parquet"
    out.to_parquet(out_path, index=False)
    logger.info("Wrote %d design pockets → %s", len(out), out_path)
    return out
