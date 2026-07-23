"""Phase 1 — catalytic pocket artifacts (source of truth for design indexing)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from catalyst_atlas.data.structures import fetch_pdb_text, parse_ca_atoms
from catalyst_atlas.paths import PROCESSED, RAW, ensure_dirs
from catalyst_atlas.site.extract import FIRST_SHELL_RADIUS

logger = logging.getLogger(__name__)

SECOND_SHELL_RADIUS = 12.0
LIGAND_CONTACT_RADIUS = 6.0


def _expand_shell_from_pdb(
    residues: list[dict[str, Any]],
    *,
    pdb_id: str,
    catalytic: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge CA atoms from the PDB into site residues out to second-shell radius."""
    if not pdb_id or not catalytic:
        return residues
    cache = RAW / "pdb"
    text = fetch_pdb_text(str(pdb_id).lower(), cache)
    if not text:
        return residues
    center = np.array([r["xyz"] for r in catalytic], dtype=float).mean(axis=0)
    by_key = {
        (str(r.get("chain") or "A"), int(r["resnum"])): r for r in residues
    }
    cat_keys = {(str(r.get("chain") or "A"), int(r["resnum"])) for r in catalytic}
    for atom in parse_ca_atoms(text):
        key = (str(atom["chain"]), int(atom["resnum"]))
        if key in cat_keys:
            continue
        d = float(np.linalg.norm(np.array(atom["xyz"], dtype=float) - center))
        if d > SECOND_SHELL_RADIUS:
            continue
        role = "first_shell" if d <= FIRST_SHELL_RADIUS else "second_shell"
        if key in by_key:
            # Prefer keeping annotated residue; refresh coords/role if useful.
            existing = by_key[key]
            if existing.get("role") == "catalytic":
                continue
            existing.setdefault("xyz", atom["xyz"])
            if existing.get("role") not in {"first_shell", "second_shell"}:
                existing["role"] = role
            continue
        rec = {
            "chain": atom["chain"],
            "resnum": atom["resnum"],
            "aa": atom["aa"],
            "role": role,
            "xyz": atom["xyz"],
        }
        residues.append(rec)
        by_key[key] = rec
    return residues


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


def _pdb_chain_map(
    pdb_id: str,
    chain: str,
) -> tuple[str, dict[int, int], list[int], str]:
    """Return (chain_sequence, resnum→seq_index, ordered_resnums, chain_id).

    Design indexing must follow the structure chain — PDB resnums are not
    UniProt offsets.
    """
    if not pdb_id:
        return "", {}, [], chain
    text = fetch_pdb_text(str(pdb_id).lower(), RAW / "pdb")
    if not text:
        return "", {}, [], chain
    atoms = [a for a in parse_ca_atoms(text) if str(a["chain"]) == str(chain)]
    if not atoms:
        # Fall back to first chain present.
        all_atoms = parse_ca_atoms(text)
        if not all_atoms:
            return "", {}, [], chain
        chain = str(all_atoms[0]["chain"])
        atoms = [a for a in all_atoms if str(a["chain"]) == chain]
    # Stable unique residues ordered by resnum.
    by_res: dict[int, str] = {}
    for a in atoms:
        by_res[int(a["resnum"])] = str(a["aa"])
    resnums = sorted(by_res)
    seq = "".join(by_res[n] for n in resnums)
    mapping = {n: i for i, n in enumerate(resnums)}
    return seq, mapping, resnums, chain


def _seq_index_for(
    resnum: int,
    sequence: str,
    *,
    resnum_to_idx: dict[int, int] | None = None,
) -> int | None:
    if resnum_to_idx is not None:
        return resnum_to_idx.get(int(resnum))
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

    When a PDB is available, ``sequence`` is the structure chain sequence and
    ``seq_index`` maps via PDB resnum (not UniProt numbering).
    """
    residues = _parse_json(row.get("site_residues_json"))
    ligands = _parse_json(row.get("ligands_json"))
    uniprot_sequence = str(row.get("sequence") or "")

    catalytic_raw = [r for r in residues if r.get("role") == "catalytic"]
    if not catalytic_raw:
        catalytic_raw = list(residues)

    # Expand first/second shell from the experimental PDB when available.
    residues = _expand_shell_from_pdb(
        list(residues),
        pdb_id=str(row.get("pdb_id") or ""),
        catalytic=catalytic_raw,
    )
    # Refresh catalytic list after expansion (unchanged, but keep order stable).
    catalytic_raw = [r for r in residues if r.get("role") == "catalytic"]
    if not catalytic_raw:
        catalytic_raw = [r for r in residues if r.get("role") != "first_shell"]

    design_chain = str((catalytic_raw[0].get("chain") if catalytic_raw else "A") or "A")
    pdb_seq, resnum_to_idx, chain_resnums, design_chain = _pdb_chain_map(
        str(row.get("pdb_id") or ""), design_chain
    )
    sequence = pdb_seq or uniprot_sequence
    index_map = resnum_to_idx if pdb_seq else None

    core = np.array([r["xyz"] for r in catalytic_raw], dtype=float)
    center = core.mean(axis=0)

    catalytic = [
        _residue_record(
            r,
            shell=None,
            seq_index=_seq_index_for(int(r["resnum"]), sequence, resnum_to_idx=index_map),
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
        # Only redesign residues on the design chain with a seq_index.
        if str(r.get("chain") or "A") != design_chain:
            continue
        seq_index = _seq_index_for(
            int(r["resnum"]), sequence, resnum_to_idx=index_map
        )
        if seq_index is None:
            continue
        redesignable.append(
            _residue_record(
                r,
                shell=shell,
                seq_index=seq_index,
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
        "design_chain": design_chain,
        "chain_resnums": chain_resnums,
        "sequence": sequence,
        "uniprot_sequence": uniprot_sequence,
        "sequence_source": "pdb_chain" if pdb_seq else "uniprot",
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
