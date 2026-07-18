"""Detect catalytic cofactors / metals from PDB HETATM near the reaction center."""

from __future__ import annotations

from typing import Any

import numpy as np

# Map common PDB hetero names → chemistry tags used in features / cards.
HET_TO_TAG: dict[str, str] = {
    # Nicotinamide
    "NAD": "NAD",
    "NAI": "NAD",
    "NAJ": "NAD",
    "NAP": "NADP",
    "NDP": "NADP",
    "NADP": "NADP",
    # Flavin
    "FAD": "FAD",
    "FMN": "FMN",
    # PLP
    "PLP": "PLP",
    "PMP": "PLP",
    "PYX": "PLP",
    # Heme
    "HEM": "heme",
    "HEA": "heme",
    "HEB": "heme",
    "HEC": "heme",
    "HEO": "heme",
    # Nucleotides
    "ATP": "ATP",
    "ADP": "ATP",
    "AMP": "ATP",
    "GTP": "GTP",
    "GDP": "GTP",
    "SAM": "SAM",
    "SAH": "SAM",
    "COA": "CoA",
    "ACO": "CoA",
    # Metals (element / common ion codes)
    "ZN": "Zn",
    "FE": "Fe",
    "FE2": "Fe",
    "FE3": "Fe",
    "MG": "Mg",
    "MN": "Mn",
    "CA": "Ca",
    "CU": "Cu",
    "CO": "Co",
    "NI": "Ni",
    "K": "K",
    "NA": "Na",
}

METAL_TAGS = {"Zn", "Fe", "Mg", "Mn", "Ca", "Cu", "Co", "Ni", "K", "Na"}

# Feature vocabulary order (must stay stable for saved matrices).
COFACTOR_VOCAB = [
    "none",
    "NAD",
    "NADP",
    "FAD",
    "FMN",
    "PLP",
    "heme",
    "ATP",
    "GTP",
    "SAM",
    "CoA",
    "Zn",
    "Fe",
    "Mg",
    "Mn",
    "Ca",
    "Cu",
    "Co",
]


def _het_centroid(lines: list[str]) -> list[float] | None:
    coords = []
    for line in lines:
        try:
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        except ValueError:
            continue
    if not coords:
        return None
    return np.mean(np.asarray(coords, dtype=float), axis=0).tolist()


def parse_hetatm_groups(pdb_text: str) -> list[dict[str, Any]]:
    """Group HETATM records by (resname, chain, resnum)."""
    groups: dict[tuple[str, str, int], list[str]] = {}
    for line in pdb_text.splitlines():
        if line.startswith("ENDMDL") and groups:
            break
        if not line.startswith("HETATM"):
            continue
        if len(line) < 54:
            continue
        resname = line[17:20].strip().upper()
        chain = line[21].strip() or "A"
        try:
            resnum = int(line[22:26])
        except ValueError:
            continue
        key = (resname, chain, resnum)
        groups.setdefault(key, []).append(line)

    out: list[dict[str, Any]] = []
    for (resname, chain, resnum), lines in groups.items():
        tag = HET_TO_TAG.get(resname)
        if tag is None:
            continue
        xyz = _het_centroid(lines)
        if xyz is None:
            continue
        out.append(
            {
                "name": tag,
                "het": resname,
                "kind": "metal" if tag in METAL_TAGS else "cofactor",
                "chain": chain,
                "resnum": resnum,
                "xyz": xyz,
            }
        )
    return out


def cofactors_near_site(
    pdb_text: str,
    catalytic: list[dict[str, Any]],
    radius: float = 8.0,
) -> tuple[list[dict[str, Any]], str]:
    """Return ligands within ``radius`` Å of the catalytic centroid + tag string."""
    if not catalytic:
        return [], "none"
    center = np.mean(np.asarray([r["xyz"] for r in catalytic], dtype=float), axis=0)
    near: list[dict[str, Any]] = []
    tags: set[str] = set()
    for lig in parse_hetatm_groups(pdb_text):
        d = float(np.linalg.norm(np.asarray(lig["xyz"], dtype=float) - center))
        if d <= radius:
            near.append({**lig, "dist_to_core": d})
            tags.add(lig["name"])
    tag_str = ",".join(sorted(tags)) if tags else "none"
    return near, tag_str
