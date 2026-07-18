"""Engineered representation of the catalytic microenvironment.

Explicitly NOT:
- whole-protein structure descriptors
- fold similarity
- pocket-shape-only features

IS:
- catalytic residue chemistry
- first-shell electrostatic / polarity composition
- catalytic pairwise geometry
- cofactor / metal flags
- ligand-contact summary
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from catalyst_atlas.paths import PROCESSED, ensure_dirs

logger = logging.getLogger(__name__)

AA20 = list("ACDEFGHIKLMNPQRSTVWY")
CHARGED = set("DEKRH")
POLAR = set("STNQYC")
HYDROPHOBIC = set("AILMFWVP")
AROMATIC = set("FYWH")
COFACTOR_VOCAB = [
    "none",
    "NAD",
    "NADP",
    "Zn",
    "Fe",
    "Mg",
    "Mn",
    "PLP",
    "ATP",
    "heme",
    "FAD",
]


def _aa_composition(aas: str) -> np.ndarray:
    counts = np.zeros(len(AA20), dtype=float)
    if not aas:
        return counts
    for ch in aas:
        if ch in AA20:
            counts[AA20.index(ch)] += 1.0
    return counts / max(len(aas), 1)


def _chem_proxy(aas: str) -> np.ndarray:
    n = max(len(aas), 1)
    return np.array(
        [
            sum(a in CHARGED for a in aas) / n,
            sum(a in POLAR for a in aas) / n,
            sum(a in HYDROPHOBIC for a in aas) / n,
            sum(a in AROMATIC for a in aas) / n,
            sum(a == "C" for a in aas) / n,
            sum(a == "H" for a in aas) / n,
            sum(a in "DE" for a in aas) / n,
            sum(a in "KR" for a in aas) / n,
        ],
        dtype=float,
    )


def _geometry_histogram(pairwise_json: str, n_bins: int = 8) -> np.ndarray:
    pairs = json.loads(pairwise_json) if pairwise_json else []
    dists = np.array([p["distance"] for p in pairs], dtype=float)
    if dists.size == 0:
        return np.zeros(n_bins + 3, dtype=float)
    hist, _ = np.histogram(dists, bins=n_bins, range=(2.0, 12.0), density=True)
    stats = np.array([dists.mean(), dists.std() if dists.size > 1 else 0.0, dists.min()])
    return np.concatenate([hist.astype(float), stats])


def _cofactor_onehot(cofactor_names: str) -> np.ndarray:
    tags = {t.strip() for t in (cofactor_names or "none").split(",") if t.strip()}
    if not tags:
        tags = {"none"}
    vec = np.zeros(len(COFACTOR_VOCAB), dtype=float)
    for i, name in enumerate(COFACTOR_VOCAB):
        if name in tags:
            vec[i] = 1.0
    return vec


def _ligand_contact_stats(ligand_contacts_json: str) -> np.ndarray:
    contacts = json.loads(ligand_contacts_json) if ligand_contacts_json else []
    if not contacts:
        return np.zeros(3, dtype=float)
    dists = np.array([c["distance"] for c in contacts], dtype=float)
    return np.array([len(contacts), dists.mean(), dists.min()], dtype=float)


def featurize_row(row: pd.Series, composition_only: bool = False) -> np.ndarray:
    cat_comp = _aa_composition(row.get("catalytic_aas", "") or "")
    shell_comp = _aa_composition(row.get("first_shell_aas", "") or "")
    if composition_only:
        return np.concatenate([cat_comp, shell_comp])

    cat_proxy = _chem_proxy(row.get("catalytic_aas", "") or "")
    shell_proxy = _chem_proxy(row.get("first_shell_aas", "") or "")
    geom = _geometry_histogram(row.get("pairwise_json", "[]") or "[]")
    cof = _cofactor_onehot(row.get("cofactor_names", "none") or "none")
    lig = _ligand_contact_stats(row.get("ligand_contacts_json", "[]") or "[]")
    sizes = np.array(
        [
            float(row.get("n_catalytic", 0) or 0),
            float(row.get("n_first_shell", 0) or 0),
            float(row.get("n_cofactors", 0) or 0),
        ],
        dtype=float,
    )
    return np.concatenate([cat_comp, shell_comp, cat_proxy, shell_proxy, geom, cof, lig, sizes])


FEATURE_NAMES_HINT = (
    "catalytic_aa_comp + first_shell_aa_comp + catalytic_chem_proxy + "
    "first_shell_chem_proxy + pairwise_geometry_hist + cofactor_onehot + "
    "ligand_contact_stats + size_counts"
)


def build_feature_matrix(
    micro_df: pd.DataFrame | None = None,
    composition_only: bool = False,
    scale: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, list[str], StandardScaler | None]:
    ensure_dirs()
    if micro_df is None:
        path = PROCESSED / "microenvironments.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run cat-sites first")
        micro_df = pd.read_parquet(path)

    X = np.vstack([featurize_row(row, composition_only=composition_only) for _, row in micro_df.iterrows()])
    ids = micro_df["enzyme_id"].tolist()
    scaler = None
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    meta = micro_df[
        [
            c
            for c in [
                "enzyme_id",
                "chemistry_class",
                "catalytic_pattern",
                "cofactor_tags",
                "substrate_class",
                "family_id",
                "seq_cluster",
                "fold_cluster",
                "ec_number",
                "sequence",
                "is_cryptic_seed",
            ]
            if c in micro_df.columns
        ]
    ].copy()

    suffix = "composition" if composition_only else "full"
    np.save(PROCESSED / f"features_{suffix}.npy", X)
    meta.to_parquet(PROCESSED / f"features_{suffix}_meta.parquet", index=False)
    logger.info(
        "Built %s feature matrix %s for %d enzymes (%s)",
        suffix,
        X.shape,
        len(ids),
        FEATURE_NAMES_HINT if not composition_only else "composition only",
    )
    return meta, X, ids, scaler


def run_featurize() -> dict[str, Any]:
    meta, X, ids, _ = build_feature_matrix(composition_only=False, scale=True)
    build_feature_matrix(composition_only=True, scale=True)
    return {"n_enzymes": len(ids), "n_features": int(X.shape[1]), "meta": meta}
