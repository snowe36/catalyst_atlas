"""Engineered catalytic embedding + retrieval-augmented chemistry readout.

v1 default: standardized microenvironment features as the embedding space,
with kNN chemistry transfer. Deep / ESM models are deferred until they earn
their place on hard holdouts.
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
    ensure_dirs()
    meta, X, _, _ = build_feature_matrix(composition_only=composition_only, scale=True)
    nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(meta)), metric="euclidean")
    nn.fit(X)
    mode = "composition" if composition_only else "full"
    # Persist embedding matrix for eval / search.
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

    neighbors = []
    votes: dict[str, float] = {}
    pattern_votes: dict[str, float] = {}
    cofactor_votes: dict[str, float] = {}
    for i, d in filtered:
        row = index.meta.iloc[i]
        weight = 1.0 / (float(d) + 1e-3)
        chem = row["chemistry_class"]
        votes[chem] = votes.get(chem, 0.0) + weight
        pat = row["catalytic_pattern"]
        pattern_votes[pat] = pattern_votes.get(pat, 0.0) + weight
        for tag in str(row.get("cofactor_tags", "none")).split(","):
            tag = tag.strip() or "none"
            cofactor_votes[tag] = cofactor_votes.get(tag, 0.0) + weight
        neighbors.append(
            {
                "enzyme_id": row["enzyme_id"],
                "chemistry_class": chem,
                "catalytic_pattern": pat,
                "cofactor_tags": row.get("cofactor_tags", "none"),
                "family_id": row.get("family_id", ""),
                "distance": float(d),
                "seq_cluster": int(row.get("seq_cluster", -1)),
                "fold_cluster": int(row.get("fold_cluster", -1)),
            }
        )

    pred_chemistry = max(votes, key=votes.get) if votes else "unknown"
    pred_pattern = max(pattern_votes, key=pattern_votes.get) if pattern_votes else "unknown"
    pred_cofactors = sorted(cofactor_votes, key=cofactor_votes.get, reverse=True)[:3]
    conf = votes.get(pred_chemistry, 0.0) / (sum(votes.values()) + 1e-9)

    query_row = index.meta.iloc[query_idx]
    return {
        "query_enzyme_id": query_row["enzyme_id"],
        "true_chemistry_class": query_row.get("chemistry_class"),
        "true_catalytic_pattern": query_row.get("catalytic_pattern"),
        "true_cofactor_tags": query_row.get("cofactor_tags"),
        "predicted_chemistry_class": pred_chemistry,
        "predicted_catalytic_pattern": pred_pattern,
        "predicted_cofactor_tags": pred_cofactors,
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
