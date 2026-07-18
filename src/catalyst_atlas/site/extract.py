"""Extract catalytic microenvironments — not whole folds or empty pocket shapes.

A microenvironment is the chemistry neighborhood:
- annotated catalytic residues
- first-shell residues near the catalytic core
- cofactors / metals
- local geometry among chemistry-participating atoms
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from catalyst_atlas.paths import PROCESSED, RAW, ensure_dirs

logger = logging.getLogger(__name__)

# First-shell cutoff around catalytic core (Å). Deliberately tight vs pocket volume.
FIRST_SHELL_RADIUS = 8.0


def _parse_json(cell: Any) -> list[dict[str, Any]]:
    if isinstance(cell, list):
        return cell
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    return json.loads(cell)


def extract_microenvironment(row: pd.Series) -> dict[str, Any]:
    residues = _parse_json(row["site_residues_json"])
    ligands = _parse_json(row["ligands_json"])

    catalytic = [r for r in residues if r.get("role") == "catalytic"]
    if not catalytic:
        # Fall back: treat all provided residues as catalytic anchors.
        catalytic = list(residues)

    core = np.array([r["xyz"] for r in catalytic], dtype=float)
    center = core.mean(axis=0)

    # Keep first-shell neighbors within radius of catalytic center.
    first_shell = []
    for r in residues:
        if r.get("role") == "catalytic":
            continue
        d = float(np.linalg.norm(np.array(r["xyz"], dtype=float) - center))
        if d <= FIRST_SHELL_RADIUS:
            first_shell.append({**r, "dist_to_core": d})

    # Pairwise catalytic geometry (the chemical machine, not fold TM-score).
    pairwise = []
    for i in range(len(catalytic)):
        for j in range(i + 1, len(catalytic)):
            a = np.array(catalytic[i]["xyz"], dtype=float)
            b = np.array(catalytic[j]["xyz"], dtype=float)
            pairwise.append(
                {
                    "i": i,
                    "j": j,
                    "aa_i": catalytic[i]["aa"],
                    "aa_j": catalytic[j]["aa"],
                    "distance": float(np.linalg.norm(a - b)),
                }
            )

    cofactors = [lig for lig in ligands if lig.get("kind") in {"cofactor", "metal"}]
    ligand_contacts = []
    for lig in cofactors:
        lxyz = np.array(lig["xyz"], dtype=float)
        for r in catalytic + first_shell:
            d = float(np.linalg.norm(np.array(r["xyz"], dtype=float) - lxyz))
            if d <= 6.0:
                ligand_contacts.append(
                    {
                        "ligand": lig["name"],
                        "residue": f"{r['aa']}{r['resnum']}",
                        "distance": d,
                    }
                )

    return {
        "enzyme_id": row["enzyme_id"],
        "n_catalytic": len(catalytic),
        "n_first_shell": len(first_shell),
        "n_cofactors": len(cofactors),
        "catalytic_aas": "".join(r["aa"] for r in catalytic),
        "first_shell_aas": "".join(r["aa"] for r in first_shell),
        "pairwise_json": json.dumps(pairwise),
        "ligand_contacts_json": json.dumps(ligand_contacts),
        "cofactor_names": ",".join(sorted({c["name"] for c in cofactors})) or "none",
        "core_centroid_json": json.dumps(center.tolist()),
        "microenvironment_json": json.dumps(
            {
                "catalytic": catalytic,
                "first_shell": first_shell,
                "ligands": cofactors,
            }
        ),
    }


def run_site_extraction(raw_path: Path | None = None) -> pd.DataFrame:
    ensure_dirs()
    path = raw_path or (RAW / "catalytic_atlas.parquet")
    if not path.exists():
        raise FileNotFoundError(f"Missing raw atlas at {path}; run cat-download first")

    atlas = pd.read_parquet(path)
    micros = [extract_microenvironment(row) for _, row in atlas.iterrows()]
    micro_df = pd.DataFrame(micros)

    # Join labels needed downstream.
    label_cols = [
        "enzyme_id",
        "uniprot_id",
        "pdb_id",
        "family_id",
        "chemistry_family",
        "mechanistic_pattern",
        "chemistry_class",
        "catalytic_pattern",
        "cofactor_tags",
        "substrate_class",
        "ec_number",
        "sequence",
        "seq_cluster",
        "fold_cluster",
        "source",
        "is_cryptic_seed",
        "enzyme_name",
        "cath_topology",
    ]
    keep = [c for c in label_cols if c in atlas.columns]
    out = micro_df.merge(atlas[keep], on="enzyme_id", how="left")

    out_path = PROCESSED / "microenvironments.parquet"
    out.to_parquet(out_path, index=False)
    logger.info("Extracted %d catalytic microenvironments → %s", len(out), out_path)
    return out
