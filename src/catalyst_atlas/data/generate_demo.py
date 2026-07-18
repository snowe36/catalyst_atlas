"""Build a high-confidence demo atlas of catalytic microenvironments.

Quality over web-scale weak labels. Enzymes are generated from curated
mechanistic family templates so that:

- members of a family share catalytic geometry / chemistry
- sequence clusters are *not* aligned with chemistry (cryptic analogs)
- some fold clusters mix chemistries (hard negatives for fold cluster-lookup)
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from catalyst_atlas.data.ontology import families
from catalyst_atlas.paths import RAW, ensure_dirs

logger = logging.getLogger(__name__)

AA = list("ACDEFGHIKLMNPQRSTVWY")
AA_INDEX = {a: i for i, a in enumerate(AA)}


def _stable_int(text: str, *, salt: int = 0, mod: int | None = None) -> int:
    """Process-stable hash (unlike Python's salted ``hash()``)."""
    digest = hashlib.blake2b(f"{salt}:{text}".encode(), digest_size=8).digest()
    value = int.from_bytes(digest, "little")
    return value % mod if mod is not None else value


def _random_sequence(rng: np.random.Generator, length: int = 280) -> str:
    return "".join(rng.choice(AA, size=length))


def _catalytic_coords(
    rng: np.random.Generator,
    residues: list[str],
    family_id: str,
    noise: float = 0.35,
) -> list[dict[str, Any]]:
    """Place catalytic residues in a family-specific geometry with small noise."""
    # Deterministic family anchor so same-family sites stay similar across processes.
    seed = _stable_int(family_id, mod=2**31)
    anchor_rng = np.random.default_rng(seed)
    anchors = anchor_rng.normal(scale=2.5, size=(len(residues), 3))
    # Enforce chemically plausible pairwise spacing (~3–8 Å).
    for i in range(1, len(anchors)):
        vec = anchors[i] - anchors[0]
        norm = np.linalg.norm(vec) + 1e-6
        target = 3.5 + 1.2 * i
        anchors[i] = anchors[0] + vec / norm * target

    coords = anchors + rng.normal(scale=noise, size=anchors.shape)
    out = []
    for i, aa in enumerate(residues):
        out.append(
            {
                "chain": "A",
                "resnum": 40 + i * 25,
                "aa": aa,
                "role": "catalytic",
                "xyz": coords[i].tolist(),
            }
        )
    return out


def _neighborhood(
    rng: np.random.Generator,
    catalytic: list[dict[str, Any]],
    n_extra: int = 8,
) -> list[dict[str, Any]]:
    """Add first-shell neighbors around the catalytic core (not whole pocket shape)."""
    core = np.array([r["xyz"] for r in catalytic], dtype=float)
    center = core.mean(axis=0)
    neighbors = []
    for j in range(n_extra):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction) + 1e-6
        radius = rng.uniform(4.0, 7.5)
        xyz = center + direction * radius
        aa = rng.choice(list("DEHKNQSTYR"))  # polar/charged-enriched first shell
        neighbors.append(
            {
                "chain": "A",
                "resnum": 200 + j,
                "aa": str(aa),
                "role": "first_shell",
                "xyz": xyz.tolist(),
            }
        )
    return neighbors


def _cofactor_site(
    family: dict[str, Any],
    catalytic: list[dict[str, Any]],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    tags = family.get("cofactor_tags") or ["none"]
    if tags == ["none"]:
        return []
    core = np.array([r["xyz"] for r in catalytic], dtype=float).mean(axis=0)
    ligands = []
    for tag in tags:
        if tag == "none":
            continue
        offset = rng.normal(scale=0.4, size=3)
        offset = offset / (np.linalg.norm(offset) + 1e-6) * rng.uniform(3.0, 5.0)
        ligands.append(
            {
                "name": tag,
                "kind": "cofactor" if tag not in {"Zn", "Fe", "Mg", "Mn"} else "metal",
                "xyz": (core + offset).tolist(),
            }
        )
    return ligands


def generate_demo_atlas(
    n_enzymes: int = 800,
    seed: int = 7,
    cryptic_fraction: float = 0.35,
) -> pd.DataFrame:
    """Generate a high-confidence catalytic atlas for offline / CI use.

    Target scale for the full public release is ~10–15k curated sites.
    The demo generator produces a smaller but label-clean atlas that
    preserves the scientific structure (cryptic analogs, fold traps).
    """
    ensure_dirs()
    rng = np.random.default_rng(seed)
    fams = families()
    rows: list[dict[str, Any]] = []

    # Pre-allocate sequence / fold cluster ids that cut across chemistry.
    n_seq_clusters = max(40, n_enzymes // 15)
    n_fold_clusters = max(20, n_enzymes // 30)

    per_family = max(1, n_enzymes // len(fams))
    enzyme_idx = 0
    for fam in fams:
        for _ in range(per_family):
            if enzyme_idx >= n_enzymes:
                break
            catalytic = _catalytic_coords(
                rng, list(fam["catalytic_residues"]), fam["id"], noise=0.25 + 0.2 * rng.random()
            )
            site_residues = catalytic + _neighborhood(rng, catalytic)
            ligands = _cofactor_site(fam, catalytic, rng)

            # Cryptic analogs: same chemistry, deliberately distant sequence cluster.
            is_cryptic = bool(rng.random() < cryptic_fraction)
            if is_cryptic:
                seq_cluster = int(rng.integers(0, n_seq_clusters))
            else:
                # Mild chemistry–sequence coupling for some members (realistic leakage).
                seq_cluster = (
                    _stable_int(fam["chemistry_class"], salt=seed) + enzyme_idx
                ) % n_seq_clusters

            # Fold traps: occasionally put different chemistries in the same fold.
            if rng.random() < 0.25:
                fold_cluster = int(rng.integers(0, n_fold_clusters))
            else:
                fold_cluster = (
                    _stable_int(fam["catalytic_pattern"], salt=seed) + enzyme_idx // 3
                ) % n_fold_clusters

            seq = _random_sequence(rng, length=int(rng.integers(220, 360)))
            # Plant catalytic residues into the sequence at annotated positions.
            seq_list = list(seq)
            for res in catalytic:
                pos = min(len(seq_list) - 1, max(0, res["resnum"] - 1))
                seq_list[pos] = res["aa"]
            seq = "".join(seq_list)

            ec = f"{fam['ec_prefix']}.{int(rng.integers(1, 9))}.{int(rng.integers(1, 20))}"
            rows.append(
                {
                    "enzyme_id": f"CAT{enzyme_idx:05d}",
                    "uniprot_id": f"DEMO{enzyme_idx:05d}",
                    "pdb_id": f"D{enzyme_idx:03d}",
                    "family_id": fam["id"],
                    "chemistry_class": fam["chemistry_class"],
                    "catalytic_pattern": fam["catalytic_pattern"],
                    "cofactor_tags": ",".join(fam.get("cofactor_tags") or ["none"]),
                    "substrate_class": fam.get("substrate_class", "unknown"),
                    "ec_number": ec,
                    "sequence": seq,
                    "seq_cluster": int(seq_cluster),
                    "fold_cluster": int(fold_cluster),
                    "site_residues_json": json.dumps(site_residues),
                    "ligands_json": json.dumps(ligands),
                    "source": "demo_high_confidence",
                    "is_cryptic_seed": is_cryptic,
                }
            )
            enzyme_idx += 1
        if enzyme_idx >= n_enzymes:
            break

    # Pad if families * per_family undershot.
    while len(rows) < n_enzymes:
        fam = fams[len(rows) % len(fams)]
        catalytic = _catalytic_coords(rng, list(fam["catalytic_residues"]), fam["id"])
        site_residues = catalytic + _neighborhood(rng, catalytic)
        ligands = _cofactor_site(fam, catalytic, rng)
        seq = _random_sequence(rng)
        rows.append(
            {
                "enzyme_id": f"CAT{len(rows):05d}",
                "uniprot_id": f"DEMO{len(rows):05d}",
                "pdb_id": f"D{len(rows):03d}",
                "family_id": fam["id"],
                "chemistry_class": fam["chemistry_class"],
                "catalytic_pattern": fam["catalytic_pattern"],
                "cofactor_tags": ",".join(fam.get("cofactor_tags") or ["none"]),
                "substrate_class": fam.get("substrate_class", "unknown"),
                "ec_number": f"{fam['ec_prefix']}.1.1",
                "sequence": seq,
                "seq_cluster": int(rng.integers(0, n_seq_clusters)),
                "fold_cluster": int(rng.integers(0, n_fold_clusters)),
                "site_residues_json": json.dumps(site_residues),
                "ligands_json": json.dumps(ligands),
                "source": "demo_high_confidence",
                "is_cryptic_seed": True,
            }
        )

    df = pd.DataFrame(rows[:n_enzymes])
    logger.info(
        "Generated demo atlas: %d enzymes, %d chemistry classes, %d patterns",
        len(df),
        df["chemistry_class"].nunique(),
        df["catalytic_pattern"].nunique(),
    )
    return df


def save_raw_atlas(df: pd.DataFrame) -> Path:
    ensure_dirs()
    out = RAW / "catalytic_atlas.parquet"
    df.to_parquet(out, index=False)
    meta = {
        "n_enzymes": int(len(df)),
        "chemistry_classes": sorted(df["chemistry_class"].unique().tolist()),
        "source": df["source"].iloc[0] if len(df) else "empty",
        "note": (
            "High-confidence catalytic annotations. Demo generator targets "
            "label quality; scale toward 10–15k curated public sites for release."
        ),
    }
    (RAW / "catalytic_atlas.meta.json").write_text(json.dumps(meta, indent=2))
    return out
