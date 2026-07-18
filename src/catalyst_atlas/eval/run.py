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
from sklearn.preprocessing import StandardScaler

from catalyst_atlas.data.cluster import pairwise_kmer_similarity_matrix
from catalyst_atlas.eval.baselines import (
    cluster_transfer,
    knn_transfer,
    same_cluster_neighbor_transfer,
    sequence_similarity_transfer,
)
from catalyst_atlas.eval.external_baselines import (
    prepare_retrieval_baselines,
    retrieval_chemistry_transfer,
)
from catalyst_atlas.eval.labels import chemistry_label_col
from catalyst_atlas.eval.metrics import accuracy, macro_f1, recall_at_k_chemistry
from catalyst_atlas.eval.splits import make_splits
from catalyst_atlas.paths import FIGURES, PROCESSED, ensure_dirs

logger = logging.getLogger(__name__)


def _load_unscaled_features() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load persisted unscaled feature matrices for leakage-aware eval."""
    full_meta_path = PROCESSED / "features_full_meta.parquet"
    full_path = PROCESSED / "features_full.npy"
    comp_path = PROCESSED / "features_composition.npy"
    comp_meta_path = PROCESSED / "features_composition_meta.parquet"
    if not full_path.exists() or not full_meta_path.exists():
        raise FileNotFoundError(
            f"Missing unscaled features under {PROCESSED}; run cat-embed (or cat-sites + featurize) first"
        )
    meta = pd.read_parquet(full_meta_path).reset_index(drop=True)
    X_full = np.load(full_path)
    if not comp_path.exists():
        raise FileNotFoundError(f"Missing {comp_path}; run cat-embed first")
    X_comp = np.load(comp_path)
    if comp_meta_path.exists():
        comp_meta = pd.read_parquet(comp_meta_path).reset_index(drop=True)
        if not meta["enzyme_id"].equals(comp_meta["enzyme_id"]):
            comp_map = {eid: i for i, eid in enumerate(comp_meta["enzyme_id"])}
            order = [comp_map[eid] for eid in meta["enzyme_id"]]
            X_comp = X_comp[order]
    return meta, X_full, X_comp


def _scale_train_test(
    X: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fit StandardScaler on train only; transform train and test."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_test = scaler.transform(X[test_idx])
    return X_train, X_test


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
    seq_sim: np.ndarray | None = None,
    label_col: str | None = None,
    mmseqs_hits: pd.DataFrame | None = None,
    foldseek_hits: pd.DataFrame | None = None,
) -> dict[str, Any]:
    label_col = label_col or chemistry_label_col(meta)
    y_train = meta.iloc[train_idx][label_col].tolist()
    y_test = meta.iloc[test_idx][label_col].tolist()

    X_full_train, X_full_test = _scale_train_test(X_full, train_idx, test_idx)
    X_comp_train, X_comp_test = _scale_train_test(X_comp, train_idx, test_idx)

    methods = {
        "catalyst_microenvironment": knn_transfer(X_full_train, y_train, X_full_test, k=k),
        "composition_only": knn_transfer(X_comp_train, y_train, X_comp_test, k=k),
        "sequence_similarity_transfer": sequence_similarity_transfer(
            meta, train_idx, test_idx, label_col=label_col, sim=seq_sim
        ),
        "sequence_cluster_transfer": same_cluster_neighbor_transfer(
            meta, train_idx, test_idx, "seq_cluster", label_col=label_col
        ),
        "fold_cluster_transfer": same_cluster_neighbor_transfer(
            meta, train_idx, test_idx, "fold_cluster", label_col=label_col
        ),
        "sequence_cluster_prior": cluster_transfer(
            meta, train_idx, test_idx, "seq_cluster", label_col=label_col
        ),
        "fold_cluster_prior": cluster_transfer(
            meta, train_idx, test_idx, "fold_cluster", label_col=label_col
        ),
    }
    if mmseqs_hits is not None and not mmseqs_hits.empty:
        methods["mmseqs_transfer"] = retrieval_chemistry_transfer(
            mmseqs_hits, meta, train_idx, test_idx, label_col=label_col
        )
    if foldseek_hits is not None and not foldseek_hits.empty:
        methods["foldseek_transfer"] = retrieval_chemistry_transfer(
            foldseek_hits, meta, train_idx, test_idx, label_col=label_col
        )

    neigh_lists = _neighbor_label_lists(X_full_train, y_train, X_full_test, k=k)
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
        "mmseqs_transfer",
        "foldseek_transfer",
        "sequence_similarity_transfer",
        "fold_cluster_transfer",
    ]
    plot_df = df[df["method"].isin(keep)].copy()
    plot_df["method"] = plot_df["method"].map(
        {
            "catalyst_microenvironment": "Catalyst Atlas",
            "mmseqs_transfer": "MMseqs2 transfer",
            "foldseek_transfer": "Foldseek transfer",
            "sequence_similarity_transfer": "Seq retrieval (k-mer NN)",
            "fold_cluster_transfer": "Fold retrieval (CATH)",
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


def run_eval(
    k: int = 5,
    test_size: float = 0.2,
    seed: int = 7,
    run_external: bool = True,
    threads: int = 4,
) -> dict[str, Any]:
    ensure_dirs()
    meta, X_full, X_comp = _load_unscaled_features()
    label_col = chemistry_label_col(meta)
    seq_sim = None
    if "sequence" in meta.columns and meta["sequence"].fillna("").str.len().gt(0).any():
        logger.info("Building k-mer sequence similarity matrix for sequence retrieval baseline")
        seq_sim = pairwise_kmer_similarity_matrix(meta["sequence"].fillna("").tolist())

    mmseqs_hits = None
    foldseek_hits = None
    external_info: dict[str, Any] = {}
    if run_external:
        logger.info("Preparing MMseqs2 / Foldseek retrieval baselines (if installed)")
        external_info = prepare_retrieval_baselines(meta, threads=threads)
        mmseqs_hits = external_info.get("mmseqs_hits")
        foldseek_hits = external_info.get("foldseek_hits")

    splits = make_splits(meta, test_size=test_size, seed=seed, label_col=label_col)
    results: dict[str, Any] = {
        "k": k,
        "label_col": label_col,
        "scaler": "StandardScaler fit on train split only",
        "external_tools": external_info.get("tools", {}),
        "splits": {},
    }
    for name, (train_idx, test_idx) in splits.items():
        logger.info("Evaluating split=%s (train=%d test=%d)", name, len(train_idx), len(test_idx))
        results["splits"][name] = evaluate_split(
            meta,
            X_full,
            X_comp,
            train_idx,
            test_idx,
            k=k,
            seq_sim=seq_sim,
            label_col=label_col,
            mmseqs_hits=mmseqs_hits,
            foldseek_hits=foldseek_hits,
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
            meta,
            X_full,
            X_comp,
            train_idx,
            np.array(cryptic_mask),
            k=k,
            seq_sim=seq_sim,
            label_col=label_col,
            mmseqs_hits=mmseqs_hits,
            foldseek_hits=foldseek_hits,
        )
        results["cryptic_seq_holdout"] = sub

    out_path = PROCESSED / "eval_metrics.json"
    out_path.write_text(json.dumps(results, indent=2))
    _plot_results(results)
    logger.info("Wrote %s and figures under %s", out_path, FIGURES)
    return results
