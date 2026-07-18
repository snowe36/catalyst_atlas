"""Baselines that transfer chemistry from nearest neighbors under different spaces."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def _vote_label(labels: list[str], weights: list[float] | None = None) -> str:
    if not labels:
        return "unknown"
    if weights is None:
        return Counter(labels).most_common(1)[0][0]
    scores: dict[str, float] = {}
    for lab, w in zip(labels, weights, strict=True):
        scores[lab] = scores.get(lab, 0.0) + w
    return max(scores, key=scores.get)


def knn_transfer(
    X_train: np.ndarray,
    y_train: list[str],
    X_test: np.ndarray,
    k: int = 5,
) -> list[str]:
    k = min(k, len(y_train))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(X_train)
    distances, indices = nn.kneighbors(X_test)
    preds = []
    for dists, inds in zip(distances, indices, strict=True):
        labels = [y_train[i] for i in inds]
        weights = [1.0 / (float(d) + 1e-3) for d in dists]
        preds.append(_vote_label(labels, weights))
    return preds


def cluster_transfer(
    meta: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    cluster_col: str,
    label_col: str = "chemistry_class",
) -> list[str]:
    """Transfer chemistry from the most common label in the nearest train cluster.

    Used as a BLAST / Foldseek *proxy* when external tools are not installed:
    - seq_cluster ≈ sequence-neighborhood transfer
    - fold_cluster ≈ fold-neighborhood transfer
    """
    train = meta.iloc[train_idx]
    cluster_to_label: dict[Any, str] = {}
    for cid, grp in train.groupby(cluster_col):
        cluster_to_label[cid] = Counter(grp[label_col]).most_common(1)[0][0]

    # Majority train label as fallback.
    fallback = Counter(train[label_col]).most_common(1)[0][0]
    preds = []
    for _, row in meta.iloc[test_idx].iterrows():
        cid = row[cluster_col]
        if cid in cluster_to_label:
            preds.append(cluster_to_label[cid])
        else:
            # Unseen cluster: use globally most common train chemistry (weak prior).
            preds.append(fallback)
    return preds


def same_cluster_neighbor_transfer(
    meta: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    cluster_col: str,
    label_col: str = "chemistry_class",
) -> list[str]:
    """If a test enzyme shares a cluster with train, transfer that cluster's chemistry.

    For unseen clusters (the hard leakage case), prediction is 'unknown' → wrong,
    which makes the baseline honestly weak — matching BLAST/Foldseek failure on
    cryptic analogs.
    """
    train = meta.iloc[train_idx]
    cluster_to_label = {
        cid: Counter(grp[label_col]).most_common(1)[0][0] for cid, grp in train.groupby(cluster_col)
    }
    preds = []
    for _, row in meta.iloc[test_idx].iterrows():
        cid = row[cluster_col]
        preds.append(cluster_to_label.get(cid, "__unseen__"))
    return preds
