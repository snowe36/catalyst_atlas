"""Lightweight PDB coordinate helpers (no Biopython required)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import requests

logger = logging.getLogger(__name__)

AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
    "MSE": "M",
}


def aa3_to_1(code: str) -> str:
    return AA3_TO_1.get(code.upper()[:3], "X")


def fetch_pdb_text(pdb_id: str, cache_dir: Path, timeout: float = 60.0) -> str | None:
    """Download a PDB file from RCSB, caching under ``cache_dir``."""
    pdb_id = pdb_id.lower()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{pdb_id}.pdb"
    if path.exists() and path.stat().st_size > 0:
        return path.read_text(errors="replace")
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code != 200 or not resp.text.lstrip().startswith(("HEADER", "TITLE", "ATOM", "MODEL")):
            logger.warning("Failed to fetch PDB %s (HTTP %s)", pdb_id, resp.status_code)
            return None
        path.write_text(resp.text)
        return resp.text
    except requests.RequestException as exc:
        logger.warning("PDB fetch error for %s: %s", pdb_id, exc)
        return None


def parse_ca_atoms(pdb_text: str) -> list[dict[str, Any]]:
    """Parse CA atoms from PDB text (first MODEL only)."""
    atoms: list[dict[str, Any]] = []
    for line in pdb_text.splitlines():
        if line.startswith("ENDMDL") and atoms:
            break
        if not line.startswith("ATOM"):
            continue
        if len(line) < 54:
            continue
        atom_name = line[12:16].strip()
        if atom_name != "CA":
            continue
        resname = line[17:20].strip()
        chain = line[21].strip() or "A"
        try:
            resnum = int(line[22:26])
        except ValueError:
            continue
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue
        atoms.append(
            {
                "chain": chain,
                "resnum": resnum,
                "aa3": resname,
                "aa": aa3_to_1(resname),
                "xyz": [x, y, z],
            }
        )
    return atoms


def build_site_from_structure(
    pdb_text: str,
    catalytic_spec: list[dict[str, Any]],
    first_shell_radius: float = 8.0,
    second_shell_radius: float = 12.0,
    cofactor_radius: float = 8.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], str]:
    """Map catalytic residues, shell neighbors, and nearby cofactors/metals.

    ``catalytic_spec`` entries need ``chain``, ``resnum``, and ``aa`` (1-letter).
    Neighbors within ``first_shell_radius`` get role ``first_shell``; those in
    (first, second] get ``second_shell``. Returns
    ``(catalytic, neighbors, ligands, cofactor_tags)``.
    """
    from catalyst_atlas.data.cofactors import cofactors_near_site

    atoms = parse_ca_atoms(pdb_text)
    by_key = {(a["chain"], a["resnum"]): a for a in atoms}

    catalytic: list[dict[str, Any]] = []
    missing = 0
    for spec in catalytic_spec:
        chain = str(spec["chain"])
        resnum = int(spec["resnum"])
        atom = by_key.get((chain, resnum))
        if atom is None:
            # Some assemblies remap chain; try any chain with that resnum + AA.
            candidates = [
                a
                for a in atoms
                if a["resnum"] == resnum and (not spec.get("aa") or a["aa"] == spec["aa"])
            ]
            atom = candidates[0] if candidates else None
        if atom is None:
            missing += 1
            continue
        catalytic.append(
            {
                "chain": atom["chain"],
                "resnum": atom["resnum"],
                "aa": atom["aa"] if atom["aa"] != "X" else spec.get("aa", "X"),
                "role": "catalytic",
                "xyz": atom["xyz"],
            }
        )

    if not catalytic:
        return [], [], [], "none"

    core = np.array([r["xyz"] for r in catalytic], dtype=float)
    center = core.mean(axis=0)
    cat_keys = {(r["chain"], r["resnum"]) for r in catalytic}

    neighbors: list[dict[str, Any]] = []
    for atom in atoms:
        key = (atom["chain"], atom["resnum"])
        if key in cat_keys:
            continue
        d = float(np.linalg.norm(np.array(atom["xyz"], dtype=float) - center))
        if d <= first_shell_radius:
            role = "first_shell"
        elif d <= second_shell_radius:
            role = "second_shell"
        else:
            continue
        neighbors.append(
            {
                "chain": atom["chain"],
                "resnum": atom["resnum"],
                "aa": atom["aa"],
                "role": role,
                "xyz": atom["xyz"],
            }
        )

    ligands, cofactor_tags = cofactors_near_site(
        pdb_text,
        catalytic,
        radius=cofactor_radius,
        site_residues=catalytic + neighbors,
    )

    if missing:
        logger.debug("Missing %d catalytic residues in structure mapping", missing)
    return catalytic, neighbors, ligands, cofactor_tags
