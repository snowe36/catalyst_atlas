"""Leakage-aware splits for chemistry identification.

Holdouts:
- random: optimistic
- seq_cluster: no shared sequence cluster between train/test
- fold_cluster: no shared fold cluster between train/test (Foldseek-hard)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def random_split(
    meta: pd.DataFrame, test_size: float = 0.2, seed: int = 7
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(len(meta))
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=meta["chemistry_class"]
    )
    return train_idx, test_idx


def group_split(
    meta: pd.DataFrame,
    group_col: str,
    test_size: float = 0.2,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    groups = meta[group_col].unique()
    rng.shuffle(groups)
    n_test = max(1, int(round(len(groups) * test_size)))
    test_groups = set(groups[:n_test])
    test_mask = meta[group_col].isin(test_groups).to_numpy()
    test_idx = np.where(test_mask)[0]
    train_idx = np.where(~test_mask)[0]
    if len(test_idx) == 0 or len(train_idx) == 0:
        return random_split(meta, test_size=test_size, seed=seed)
    return train_idx, test_idx


def make_splits(meta: pd.DataFrame, test_size: float = 0.2, seed: int = 7) -> dict[str, tuple]:
    return {
        "random": random_split(meta, test_size=test_size, seed=seed),
        "seq_cluster": group_split(meta, "seq_cluster", test_size=test_size, seed=seed),
        "fold_cluster": group_split(meta, "fold_cluster", test_size=test_size, seed=seed),
    }
