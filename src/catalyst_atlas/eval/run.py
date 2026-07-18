"""Chemistry-identification evaluation with leakage-aware splits."""

from __future__ import annotations

import json
import logging
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neighbors import NearestNeighbors

from catalyst_atlas.eval.baselines import (
    cluster_transfer,
    knn_transfer,
    same_cluster_neighbor_transfer,
)
from catalyst_atlas.eval.metrics import accuracy, macro_f1, recall_at_k_chemistry
from catalyst_atlas.eval.splits import make_splits
from catalyst_atlas.models.embed import load_index
from catalyst_atlas.paths import FIGURES, PROCESSED, ensure_dirs

logger = logging.getLogger(__name__)


def _neighbor_label_lists(
    X_train: np.ndarray,
    y_train: list[str],
    X_test: np.ndarray,
    k: int = 5,
) -> list[list[str]]:
    k = min(k, len(y_train))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(X_train)
    _, indices = nn.kneighbors(X_test)
    return [[y_train[i] for i in inds] for inds in indices]


def evaluate_split(
    meta: pd.DataFrame,
    X_full: np.ndarray,
    X_comp: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    k: int = 5,
) -> dict[str, Any]:
    y_train = meta.iloc[train_idx]["chemistry_class"].tolist()
    y_test = meta.iloc[test_idx]["chemistry_class"].tolist()

    methods = {
        "catalyst_microenvironment": knn_transfer(
            X_full[train_idx], y_train, X_full[test_idx], k=k
        ),
        "composition_only": knn_transfer(X_comp[train_idx], y_train, X_comp[test_idx], k=k),
        "sequence_cluster_transfer": same_cluster_neighbor_transfer(
            meta, train_idx, test_idx, "seq_cluster"
        ),
        "fold_cluster_transfer": same_cluster_neighbor_transfer(
            meta, train_idx, test_idx, "fold_cluster"
        ),
        "sequence_cluster_prior": cluster_transfer(meta, train_idx, test_idx, "seq_cluster"),
        "fold_cluster_prior": cluster_transfer(meta, train_idx, test_idx, "fold_cluster"),
    }

    neigh_lists = _neighbor_label_lists(X_full[train_idx], y_train, X_full[test_idx], k=k)
    out: dict[str, Any] = {
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "recall_at_k_chemistry": recall_at_k_chemistry(neigh_lists, y_test, k=k),
        "methods": {},
    }
    for name, preds in methods.items():
        # Unseen cluster sentinel counts as wrong.
        clean_preds = ["__miss__" if p == "__unseen__" else p for p in preds]
        out["methods"][name] = {
            "accuracy": accuracy(y_test, clean_preds),
            "macro_f1": macro_f1(y_test, clean_preds),
        }
    return out


def _plot_results(results: dict[str, Any]) -> None:
    ensure_dirs()
    rows = []
    for split, payload in results["splits"].items():
        for method, scores in payload["methods"].items():
            rows.append(
                {
                    "split": split,
                    "method": method,
                    "accuracy": scores["accuracy"],
                    "macro_f1": scores["macro_f1"],
                }
            )
    df = pd.DataFrame(rows)
    # Focus plot on the distinctive comparison.
    keep = [
        "catalyst_microenvironment",
        "composition_only",
        "sequence_cluster_transfer",
        "fold_cluster_transfer",
    ]
    plot_df = df[df["method"].isin(keep)].copy()
    plot_df["method"] = plot_df["method"].map(
        {
            "catalyst_microenvironment": "Catalyst microenvironment",
            "composition_only": "Composition only",
            "sequence_cluster_transfer": "Sequence cluster (BLAST proxy)",
            "fold_cluster_transfer": "Fold cluster (Foldseek proxy)",
        }
    )

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(data=plot_df, x="split", y="accuracy", hue="method", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Holdout split")
    ax.set_ylabel("Chemistry accuracy")
    ax.set_title("Chemistry ID under leakage-aware splits")
    ax.legend(title="", loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_chemistry_leakage.png", dpi=160)
    plt.close(fig)


def run_eval(k: int = 5, test_size: float = 0.2, seed: int = 7) -> dict[str, Any]:
    ensure_dirs()
    full = load_index(composition_only=False)
    comp = load_index(composition_only=True)
    meta = full.meta.reset_index(drop=True)
    # Align composition matrix to the same enzyme order.
    comp_meta = comp.meta.reset_index(drop=True)
    if not meta["enzyme_id"].equals(comp_meta["enzyme_id"]):
        comp_map = {eid: i for i, eid in enumerate(comp_meta["enzyme_id"])}
        order = [comp_map[eid] for eid in meta["enzyme_id"]]
        X_comp = comp.X[order]
    else:
        X_comp = comp.X

    splits = make_splits(meta, test_size=test_size, seed=seed)
    results: dict[str, Any] = {"k": k, "splits": {}}
    for name, (train_idx, test_idx) in splits.items():
        logger.info("Evaluating split=%s (train=%d test=%d)", name, len(train_idx), len(test_idx))
        results["splits"][name] = evaluate_split(
            meta, full.X, X_comp, train_idx, test_idx, k=k
        )

    # Cryptic-analog diagnostic: test enzymes whose seq_cluster is unseen in train
    # but whose chemistry is recoverable from microenvironment neighbors.
    train_idx, test_idx = splits["seq_cluster"]
    train_clusters = set(meta.iloc[train_idx]["seq_cluster"])
    cryptic_mask = [
        i
        for i in test_idx
        if meta.iloc[i]["seq_cluster"] not in train_clusters
        or bool(meta.iloc[i].get("is_cryptic_seed", False))
    ]
    if cryptic_mask:
        sub = evaluate_split(
            meta, full.X, X_comp, train_idx, np.array(cryptic_mask), k=k
        )
        results["cryptic_seq_holdout"] = sub

    out_path = PROCESSED / "eval_metrics.json"
    out_path.write_text(json.dumps(results, indent=2))
    _plot_results(results)
    logger.info("Wrote %s and figures under %s", out_path, FIGURES)
    return results
