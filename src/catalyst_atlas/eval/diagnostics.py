"""Secondary analyses that sharpen the chemistry-vs-evolution story.

1. Stratify chemistry-transfer accuracy by nearest-train sequence identity
2. Audit same-fold / different-chemistry traps and different-fold / same-chemistry recovery
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from catalyst_atlas.eval.metrics import accuracy, stratified_accuracy

IDENTITY_BINS: list[tuple[str, float, float]] = [
    (">80%", 80.0, 100.01),
    ("40–80%", 40.0, 80.0),
    ("20–40%", 20.0, 40.0),
    ("<20%", 0.0, 20.0),
]


def identity_bin(pct: float) -> str:
    for name, lo, hi in IDENTITY_BINS:
        if lo <= pct < hi:
            return name
    return "<20%"


def nearest_train_sequence_identity(
    meta: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    mmseqs_hits: pd.DataFrame | None = None,
    seq_sim: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    """Nearest-train sequence identity (%) for each test enzyme.

    Prefers MMseqs2 ``pident`` when hit tables are available; otherwise falls
    back to 100 × max k-mer Jaccard to train (proxy, not true %id).
    """
    n_test = len(test_idx)
    ident = np.full(n_test, 0.0, dtype=float)
    source = "none"

    train_ids = set(meta.iloc[train_idx]["enzyme_id"].astype(str))
    test_ids = [str(meta.iloc[int(i)]["enzyme_id"]) for i in test_idx]
    id_to_test_pos = {eid: p for p, eid in enumerate(test_ids)}

    if mmseqs_hits is not None and not mmseqs_hits.empty and "pident" in mmseqs_hits.columns:
        source = "mmseqs_pident"
        for _, row in mmseqs_hits.iterrows():
            q, t = str(row["query"]), str(row["target"])
            if q not in id_to_test_pos or t not in train_ids:
                continue
            pid = float(row["pident"])
            pos = id_to_test_pos[q]
            if pid > ident[pos]:
                ident[pos] = pid
        return ident, source

    if seq_sim is not None:
        source = "kmer_jaccard_proxy"
        for pos, ti in enumerate(test_idx):
            sims = seq_sim[int(ti), train_idx]
            ident[pos] = float(np.max(sims) * 100.0) if len(sims) else 0.0
        return ident, source

    return ident, source


def sequence_identity_stratified_transfer(
    y_true: list[str],
    method_preds: dict[str, list[str]],
    nearest_identity: np.ndarray,
) -> dict[str, Any]:
    """Per-bin chemistry accuracy for each method."""
    strata = [identity_bin(float(x)) for x in nearest_identity]
    bin_counts = {name: strata.count(name) for name, _, _ in IDENTITY_BINS}
    methods_out: dict[str, dict[str, float]] = {}
    for method, preds in method_preds.items():
        clean = ["__miss__" if p == "__unseen__" else p for p in preds]
        methods_out[method] = stratified_accuracy(y_true, clean, strata)
    return {
        "identity_source": None,  # filled by caller
        "bin_counts": bin_counts,
        "methods": methods_out,
    }


def _clean_preds(preds: list[str]) -> list[str]:
    return ["__miss__" if p == "__unseen__" else p for p in preds]


def fold_chemistry_audits(
    meta: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    method_preds: dict[str, list[str]],
    label_col: str = "chemistry_family",
) -> dict[str, Any]:
    """Biological wow-tests: same-fold traps and different-fold chemistry recovery."""
    train = meta.iloc[train_idx]
    test = meta.iloc[test_idx]
    y_true = test[label_col].astype(str).tolist()

    train_folds = set(train["fold_cluster"].tolist())
    # fold -> set of chemistries in train
    fold_chems: dict[Any, set[str]] = {}
    for fold, grp in train.groupby("fold_cluster"):
        fold_chems[fold] = set(grp[label_col].astype(str))

    same_fold_diff_chem_idx: list[int] = []
    diff_fold_same_chem_idx: list[int] = []

    train_chemistries = set(train[label_col].astype(str))

    for local_i, (_, row) in enumerate(test.iterrows()):
        fold = row["fold_cluster"]
        chem = str(row[label_col])
        train_chems_here = fold_chems.get(fold, set())
        # Same fold present in train, but train examples of that fold include
        # chemistry different from the query (trap for fold-based transfer).
        if fold in train_folds and train_chems_here and (
            chem not in train_chems_here or len(train_chems_here) > 1
        ):
            # Stricter trap: fold exists in train AND at least one train chem ≠ query chem
            if any(c != chem for c in train_chems_here):
                same_fold_diff_chem_idx.append(local_i)
        # No shared fold in train, but the true chemistry exists under other folds.
        if fold not in train_folds and chem in train_chemistries:
            diff_fold_same_chem_idx.append(local_i)

    def _subset_scores(indices: list[int]) -> dict[str, Any]:
        if not indices:
            return {"n": 0, "methods": {}}
        yt = [y_true[i] for i in indices]
        out_methods: dict[str, dict[str, float]] = {}
        for name, preds in method_preds.items():
            clean = _clean_preds(preds)
            yp = [clean[i] for i in indices]
            out_methods[name] = {"accuracy": accuracy(yt, yp)}
        return {"n": len(indices), "methods": out_methods}

    return {
        "same_fold_different_chemistry": {
            "description": (
                "Test enzymes whose fold appears in train with at least one "
                "different chemistry — false functional transfer trap."
            ),
            **_subset_scores(same_fold_diff_chem_idx),
        },
        "different_fold_same_chemistry": {
            "description": (
                "Test enzymes with no train fold neighbor, but whose chemistry "
                "exists under other folds — convergent chemistry recovery."
            ),
            **_subset_scores(diff_fold_same_chem_idx),
        },
    }
