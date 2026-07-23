"""Catalytic geometry from AF/ColabFold PDBs (pairwise distances vs pocket reference)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from catalyst_atlas.design.pocket import load_pocket
from catalyst_atlas.design.predict import load_prediction_metrics, write_prediction_metrics
from catalyst_atlas.paths import PROCESSED

logger = logging.getLogger(__name__)

COLABFOLD_OUT = PROCESSED / "design" / "colabfold_out"


def parse_ca_coords(pdb_path: Path) -> dict[int, np.ndarray]:
    """Map PDB residue number → CA xyz (first model / chain only)."""
    coords: dict[int, np.ndarray] = {}
    for line in pdb_path.read_text(errors="replace").splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[12:16].strip() != "CA":
            continue
        try:
            resnum = int(line[22:26])
            xyz = np.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                dtype=float,
            )
        except ValueError:
            continue
        # Keep first CA per residue (ignore altlocs / later chains).
        if resnum not in coords:
            coords[resnum] = xyz
    return coords


def catalytic_af_resnums(pocket: dict[str, Any]) -> list[int]:
    """ColabFold numbers residues 1..L in FASTA order (= seq_index + 1)."""
    out: list[int] = []
    for r in pocket.get("catalytic_residues") or []:
        if r.get("seq_index") is not None:
            out.append(int(r["seq_index"]) + 1)
        else:
            out.append(int(r["resnum"]))
    return out


def geometry_vector_from_coords(coords: list[np.ndarray]) -> list[float]:
    """Catalytic pairwise CA distances (same order as reference_geometry_vector pairs)."""
    pairs: list[float] = []
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            pairs.append(float(np.linalg.norm(coords[i] - coords[j])))
    return pairs


def geometry_vector_from_pdb(pdb_path: Path, pocket: dict[str, Any]) -> list[float] | None:
    """Build catalytic pairwise distance vector from an AF PDB + pocket."""
    ca = parse_ca_coords(pdb_path)
    if not ca:
        return None
    resnums = catalytic_af_resnums(pocket)
    coords: list[np.ndarray] = []
    for rn in resnums:
        if rn not in ca:
            logger.warning("missing CA res %s in %s", rn, pdb_path.name)
            return None
        coords.append(ca[rn])
    if len(coords) < 2:
        return None
    return geometry_vector_from_coords(coords)


def find_colabfold_pdb(enzyme_id: str, design_id: str) -> Path | None:
    """Locate unrelaxed rank-1 PDB for enzyme|design under colabfold_out."""
    if not COLABFOLD_OUT.exists():
        return None
    # Prefer metrics pdb_path if present and exists.
    metrics = load_prediction_metrics(enzyme_id, design_id)
    if metrics and metrics.get("pdb_path"):
        p = Path(str(metrics["pdb_path"]))
        if p.exists():
            return p
        # Pod path → local colabfold_out basename
        local = COLABFOLD_OUT / p.name
        if local.exists():
            return local

    stem = f"{enzyme_id}_{design_id}"
    patterns = [
        f"{stem}_unrelaxed_rank_001*.pdb",
        f"{stem}*_unrelaxed_rank_001*.pdb",
        f"{stem}*.pdb",
    ]
    for pat in patterns:
        hits = sorted(COLABFOLD_OUT.glob(pat))
        if hits:
            return hits[0]
    # ColabFold may use enzyme_design with nested design ids
    loose = sorted(COLABFOLD_OUT.glob(f"{enzyme_id}*{design_id}*.pdb"))
    return loose[0] if loose else None


def reference_catalytic_pair_vector(pocket: dict[str, Any]) -> np.ndarray:
    """Pocket catalytic pairwise distances only (ignore ligands — AF has none)."""
    cat = pocket.get("catalytic_residues") or []
    coords = [np.array(r["xyz"], dtype=float) for r in cat]
    vec = geometry_vector_from_coords(coords)
    if not vec:
        return np.zeros(1, dtype=float)
    return np.array(vec, dtype=float)


def enrich_metrics_geometry(
    enzyme_id: str,
    design_id: str,
    *,
    pdb_path: Path | None = None,
) -> dict[str, Any] | None:
    """Write geometry_vector into predictions/*/metrics.json from AF PDB."""
    pocket = load_pocket(enzyme_id)
    pdb = pdb_path or find_colabfold_pdb(enzyme_id, design_id)
    if pdb is None or not pdb.exists():
        logger.warning("no PDB for %s/%s", enzyme_id, design_id)
        return None
    vec = geometry_vector_from_pdb(pdb, pocket)
    if vec is None:
        return None

    existing = load_prediction_metrics(enzyme_id, design_id) or {}
    mean_plddt = float(existing.get("mean_plddt") or _mean_plddt_from_pdb(pdb) or 50.0)
    meta = {
        k: v
        for k, v in existing.items()
        if k
        not in {
            "enzyme_id",
            "design_id",
            "mean_plddt",
            "pocket_pae",
            "pdb_path",
            "geometry_vector",
            "geometry_source",
            "geometry_n_pairs",
        }
    }
    meta["geometry_vector"] = vec
    meta["geometry_source"] = "af_ca_pairwise"
    meta["geometry_n_pairs"] = len(vec)
    if "source" not in meta:
        meta["source"] = "colabfold"
    write_prediction_metrics(
        enzyme_id,
        design_id,
        mean_plddt=mean_plddt,
        pocket_pae=existing.get("pocket_pae"),
        pdb_path=pdb,
        meta=meta,
    )
    return load_prediction_metrics(enzyme_id, design_id)


def _mean_plddt_from_pdb(pdb_path: Path) -> float | None:
    vals: list[float] = []
    for line in pdb_path.read_text(errors="replace").splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            try:
                vals.append(float(line[60:66]))
            except ValueError:
                continue
    if not vals:
        return None
    return sum(vals) / len(vals)


def backfill_af_queue_geometry(*, queue_path: Path | None = None) -> dict[str, int]:
    """Enrich WT + AF-queue designs with geometry_vector from colabfold_out PDBs."""
    import pandas as pd

    path = queue_path or (PROCESSED / "design" / "af_queue.parquet")
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    queue = pd.read_parquet(path)
    ok = fail = 0
    seen: set[tuple[str, str]] = set()
    for _, row in queue.iterrows():
        eid = str(row["enzyme_id"])
        did = str(row["design_id"])
        key = (eid, did)
        if key in seen:
            continue
        seen.add(key)
        if enrich_metrics_geometry(eid, did) is not None:
            ok += 1
        else:
            fail += 1
    # Ensure WT rows present even if only designs are in non-wt filter
    for eid in sorted(queue["enzyme_id"].unique()):
        key = (str(eid), "WT")
        if key in seen:
            continue
        if enrich_metrics_geometry(str(eid), "WT") is not None:
            ok += 1
        else:
            fail += 1
    summary = {"n_ok": ok, "n_fail": fail, "n_unique": len(seen)}
    out = PROCESSED / "design" / "af_geometry_backfill.json"
    out.write_text(json.dumps(summary, indent=2))
    logger.info("AF geometry backfill %s → %s", summary, out)
    return summary


def purge_mock_prediction_trees() -> int:
    """Remove CAT*/mock prediction dirs so audits only see real M-CSA AF metrics."""
    root = PROCESSED / "design" / "predictions"
    if not root.exists():
        return 0
    n = 0
    for child in list(root.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        if name.startswith("CAT") or name.lower() == "mock":
            import shutil

            shutil.rmtree(child)
            n += 1
            continue
        # nested mock design ids
        for sub in list(child.iterdir()):
            if sub.is_dir() and "mock" in sub.name.lower():
                import shutil

                shutil.rmtree(sub)
                n += 1
    return n
