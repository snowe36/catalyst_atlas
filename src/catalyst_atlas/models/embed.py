"""Engineered catalytic embedding + retrieval-augmented chemistry readout.

Default: standardized microenvironment features as the embedding space,
with kNN chemistry transfer.

v0.3 optional tracks (install ``.[gpu]``):
- ``cat-graphs`` + ``cat-train-encoder`` → learned reaction-center encoder
- ``cat-esm`` → frozen ESM-2 control
Both plug into ``cat-eval`` only when their ``embedding_*.npy`` artifacts exist.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from catalyst_atlas.featurize.features import build_feature_matrix
from catalyst_atlas.paths import PROCESSED, ensure_dirs

logger = logging.getLogger(__name__)


def _query_metal_coordination(query_row: pd.Series) -> list[dict[str, Any]]:
    """Pull metal coordination shells from microenvironment JSON when present."""
    raw = query_row.get("microenvironment_json")
    if not raw:
        # Fall back: try loading from processed microenvironments by enzyme id
        return []
    try:
        micro = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return []
    out = []
    for lig in micro.get("ligands") or []:
        if lig.get("kind") != "metal":
            continue
        coord = lig.get("coordination") or {}
        out.append(
            {
                "metal": lig.get("name"),
                "geometry": coord.get("geometry"),
                "motif": coord.get("motif"),
                "n_coord": coord.get("n_coord"),
                "min_distance": coord.get("min_distance"),
            }
        )
    return out


@dataclass
class CatalogIndex:
    meta: pd.DataFrame
    X: np.ndarray
    nn: NearestNeighbors
    feature_mode: str = "full"

    def query(self, vectors: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        k = min(k, len(self.meta))
        distances, indices = self.nn.kneighbors(vectors, n_neighbors=k)
        return distances, indices


def build_index(composition_only: bool = False, n_neighbors: int = 25) -> CatalogIndex:
    """Build a retrieval index over the full catalog.

    Catalog-wide scaling is fine for search (no held-out claim). Evaluation
    loads unscaled ``features_*.npy`` and fits scalers on train only.
    """
    ensure_dirs()
    meta, X, _, _ = build_feature_matrix(composition_only=composition_only, scale=True)
    nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(meta)), metric="euclidean")
    nn.fit(X)
    mode = "composition" if composition_only else "full"
    # Persist catalog-scaled embeddings for search / chemistry cards.
    np.save(PROCESSED / f"embedding_{mode}.npy", X)
    meta.to_parquet(PROCESSED / f"embedding_{mode}_meta.parquet", index=False)
    logger.info("Built %s catalytic index over %d enzymes", mode, len(meta))
    return CatalogIndex(meta=meta, X=X, nn=nn, feature_mode=mode)


def load_index(composition_only: bool = False) -> CatalogIndex:
    mode = "composition" if composition_only else "full"
    X_path = PROCESSED / f"embedding_{mode}.npy"
    meta_path = PROCESSED / f"embedding_{mode}_meta.parquet"
    if not X_path.exists() or not meta_path.exists():
        return build_index(composition_only=composition_only)
    X = np.load(X_path)
    meta = pd.read_parquet(meta_path)
    nn = NearestNeighbors(n_neighbors=min(25, len(meta)), metric="euclidean")
    nn.fit(X)
    return CatalogIndex(meta=meta, X=X, nn=nn, feature_mode=mode)


def transfer_chemistry(
    index: CatalogIndex,
    query_idx: int,
    k: int = 5,
    exclude_self: bool = True,
) -> dict[str, Any]:
    """Retrieval-augmented chemistry card for one catalog enzyme."""
    q = index.X[query_idx : query_idx + 1]
    distances, indices = index.query(q, k=k + (1 if exclude_self else 0))
    neigh_idx = indices[0].tolist()
    neigh_dist = distances[0].tolist()
    if exclude_self:
        filtered = [(i, d) for i, d in zip(neigh_idx, neigh_dist, strict=True) if i != query_idx]
        filtered = filtered[:k]
    else:
        filtered = list(zip(neigh_idx, neigh_dist, strict=True))[:k]

    label_col = (
        "chemistry_family"
        if "chemistry_family" in index.meta.columns
        and index.meta["chemistry_family"].notna().any()
        else "chemistry_class"
    )
    neighbors = []
    votes: dict[str, float] = {}
    mech_votes: dict[str, float] = {}
    pattern_votes: dict[str, float] = {}
    cofactor_votes: dict[str, float] = {}
    for i, d in filtered:
        row = index.meta.iloc[i]
        weight = 1.0 / (float(d) + 1e-3)
        chem = row[label_col]
        votes[chem] = votes.get(chem, 0.0) + weight
        mech = row.get("mechanistic_pattern") or "unknown"
        mech_votes[str(mech)] = mech_votes.get(str(mech), 0.0) + weight
        pat = row.get("catalytic_pattern") or "unknown"
        pattern_votes[str(pat)] = pattern_votes.get(str(pat), 0.0) + weight
        for tag in str(row.get("cofactor_tags", "none")).split(","):
            tag = tag.strip() or "none"
            cofactor_votes[tag] = cofactor_votes.get(tag, 0.0) + weight
        neighbors.append(
            {
                "enzyme_id": row["enzyme_id"],
                "chemistry_family": row.get("chemistry_family", chem),
                "chemistry_class": row.get("chemistry_class", chem),
                "mechanistic_pattern": mech,
                "catalytic_pattern": pat,
                "cofactor_tags": row.get("cofactor_tags", "none"),
                "family_id": row.get("family_id", ""),
                "distance": float(d),
                "seq_cluster": int(row.get("seq_cluster", -1)),
                "fold_cluster": int(row.get("fold_cluster", -1)),
            }
        )

    pred_chemistry = max(votes, key=votes.get) if votes else "unknown"
    pred_mech = max(mech_votes, key=mech_votes.get) if mech_votes else "unknown"
    pred_pattern = max(pattern_votes, key=pattern_votes.get) if pattern_votes else "unknown"
    pred_cofactors = sorted(cofactor_votes, key=cofactor_votes.get, reverse=True)[:3]
    conf = votes.get(pred_chemistry, 0.0) / (sum(votes.values()) + 1e-9)

    query_row = index.meta.iloc[query_idx]
    true_chem = query_row.get("chemistry_family", query_row.get("chemistry_class"))
    metal_coordination = _query_metal_coordination(query_row)
    return {
        "query_enzyme_id": query_row["enzyme_id"],
        "query_enzyme_name": query_row.get("enzyme_name", ""),
        "query_fold_cluster": int(query_row.get("fold_cluster", -1)),
        "query_seq_cluster": int(query_row.get("seq_cluster", -1)),
        "label_col": label_col,
        "true_chemistry_family": true_chem,
        "true_chemistry_class": query_row.get("chemistry_class"),
        "true_mechanistic_pattern": query_row.get("mechanistic_pattern"),
        "true_catalytic_pattern": query_row.get("catalytic_pattern"),
        "true_cofactor_tags": query_row.get("cofactor_tags"),
        "predicted_chemistry_family": pred_chemistry,
        "predicted_chemistry_class": pred_chemistry,  # alias for older cards/tests
        "predicted_mechanistic_pattern": pred_mech,
        "predicted_catalytic_pattern": pred_pattern,
        "predicted_cofactor_tags": pred_cofactors,
        "metal_coordination": metal_coordination,
        "confidence": float(conf),
        "neighbors": neighbors,
        "feature_mode": index.feature_mode,
    }


def run_embed() -> dict[str, Any]:
    full = build_index(composition_only=False)
    comp = build_index(composition_only=True)
    summary = {
        "n_enzymes": len(full.meta),
        "n_features_full": int(full.X.shape[1]),
        "n_features_composition": int(comp.X.shape[1]),
    }
    (PROCESSED / "embed_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
