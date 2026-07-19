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
from catalyst_atlas.eval.diagnostics import (
    IDENTITY_BINS,
    annotation_style_audits,
    fold_chemistry_audits,
    nearest_train_sequence_identity,
    sequence_identity_stratified_transfer,
)
from catalyst_atlas.eval.external_baselines import (
    prepare_retrieval_baselines,
    retrieval_chemistry_transfer,
)
from catalyst_atlas.eval.labels import chemistry_label_col
from catalyst_atlas.eval.metrics import accuracy, macro_f1, mrr_chemistry, recall_at_k_chemistry
from catalyst_atlas.eval.splits import make_splits
from catalyst_atlas.paths import FIGURES, PROCESSED, ensure_dirs

logger = logging.getLogger(__name__)


def _align_embedding(
    meta: pd.DataFrame, emb_path, meta_path
) -> np.ndarray | None:
    """Load an embedding matrix aligned to ``meta`` enzyme_id order, or None."""
    if not emb_path.exists():
        return None
    X = np.load(emb_path)
    if meta_path.exists():
        emb_meta = pd.read_parquet(meta_path).reset_index(drop=True)
        if not meta["enzyme_id"].astype(str).equals(emb_meta["enzyme_id"].astype(str)):
            emb_map = {str(eid): i for i, eid in enumerate(emb_meta["enzyme_id"])}
            try:
                order = [emb_map[str(eid)] for eid in meta["enzyme_id"]]
            except KeyError:
                logger.warning("Could not align %s to feature meta; skipping", emb_path.name)
                return None
            X = X[order]
    if len(X) != len(meta):
        logger.warning(
            "Embedding %s length %d != meta %d; skipping", emb_path.name, len(X), len(meta)
        )
        return None
    return X


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


def _load_optional_learned_embeddings(
    meta: pd.DataFrame,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Frozen ESM + learned / fusion / ESM+GNN (+ random-graph ablation) if present."""
    X_esm = _align_embedding(
        meta,
        PROCESSED / "embedding_esm.npy",
        PROCESSED / "embedding_esm_meta.parquet",
    )
    X_learned = _align_embedding(
        meta,
        PROCESSED / "embedding_learned.npy",
        PROCESSED / "embedding_learned_meta.parquet",
    )
    X_fusion = _align_embedding(
        meta,
        PROCESSED / "embedding_fusion.npy",
        PROCESSED / "embedding_fusion_meta.parquet",
    )
    X_esm_gnn = _align_embedding(
        meta,
        PROCESSED / "embedding_esm_gnn.npy",
        PROCESSED / "embedding_esm_gnn_meta.parquet",
    )
    X_esm_gnn_rand = _align_embedding(
        meta,
        PROCESSED / "embedding_esm_gnn_randnodes.npy",
        PROCESSED / "embedding_esm_gnn_randnodes_meta.parquet",
    )
    if X_esm is not None:
        logger.info("Loaded ESM control embeddings %s", X_esm.shape)
    if X_learned is not None:
        logger.info("Loaded learned catalytic embeddings %s", X_learned.shape)
    if X_fusion is not None:
        logger.info("Loaded fusion catalytic embeddings %s", X_fusion.shape)
    if X_esm_gnn is not None:
        logger.info("Loaded ESM+GNN fusion embeddings %s", X_esm_gnn.shape)
    if X_esm_gnn_rand is not None:
        logger.info("Loaded ESM+random-graph ablation embeddings %s", X_esm_gnn_rand.shape)
    return X_esm, X_learned, X_fusion, X_esm_gnn, X_esm_gnn_rand


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


def _method_predictions(
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
    X_esm: np.ndarray | None = None,
    X_learned: np.ndarray | None = None,
    X_fusion: np.ndarray | None = None,
    X_esm_gnn: np.ndarray | None = None,
    X_esm_gnn_rand: np.ndarray | None = None,
) -> tuple[list[str], dict[str, list[str]], np.ndarray, np.ndarray]:
    """Return y_test, method→preds, and scaled full train/test matrices."""
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
    # Optional GPU / learned tracks (present only when artifacts exist).
    if X_esm is not None:
        esm_tr, esm_te = _scale_train_test(X_esm, train_idx, test_idx)
        methods["esm2_transfer"] = knn_transfer(esm_tr, y_train, esm_te, k=k)
    if X_learned is not None:
        # Already L2-normalized chemistry space — do not re-scale.
        methods["learned_catalytic_encoder"] = knn_transfer(
            X_learned[train_idx], y_train, X_learned[test_idx], k=k
        )
        # Eval-time hybrid: z-score both blocks on train, then concat.
        eng_tr, eng_te = _scale_train_test(X_full, train_idx, test_idx)
        lrn_tr, lrn_te = _scale_train_test(X_learned, train_idx, test_idx)
        Xh_tr = np.hstack([eng_tr, lrn_tr])
        Xh_te = np.hstack([eng_te, lrn_te])
        methods["catalyst_hybrid"] = knn_transfer(Xh_tr, y_train, Xh_te, k=k)
    if X_fusion is not None:
        methods["learned_fusion_encoder"] = knn_transfer(
            X_fusion[train_idx], y_train, X_fusion[test_idx], k=k
        )
    if X_esm_gnn is not None:
        methods["esm_gnn_fusion"] = knn_transfer(
            X_esm_gnn[train_idx], y_train, X_esm_gnn[test_idx], k=k
        )
    if X_esm_gnn_rand is not None:
        methods["esm_gnn_random_graph"] = knn_transfer(
            X_esm_gnn_rand[train_idx], y_train, X_esm_gnn_rand[test_idx], k=k
        )
    return y_test, methods, X_full_train, X_full_test


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
    X_esm: np.ndarray | None = None,
    X_learned: np.ndarray | None = None,
    X_fusion: np.ndarray | None = None,
    X_esm_gnn: np.ndarray | None = None,
    X_esm_gnn_rand: np.ndarray | None = None,
) -> dict[str, Any]:
    label_col = label_col or chemistry_label_col(meta)
    y_train = meta.iloc[train_idx][label_col].tolist()
    y_test, methods, X_full_train, X_full_test = _method_predictions(
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
        X_esm=X_esm,
        X_learned=X_learned,
        X_fusion=X_fusion,
        X_esm_gnn=X_esm_gnn,
        X_esm_gnn_rand=X_esm_gnn_rand,
    )

    neigh_lists = _neighbor_label_lists(X_full_train, y_train, X_full_test, k=k)
    out: dict[str, Any] = {
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "recall_at_k_chemistry": recall_at_k_chemistry(neigh_lists, y_test, k=k),
        "mrr_chemistry": mrr_chemistry(neigh_lists, y_test),
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
        "catalyst_hybrid",
        "esm_gnn_fusion",
        "esm_gnn_random_graph",
        "learned_fusion_encoder",
        "learned_catalytic_encoder",
        "esm2_transfer",
        "mmseqs_transfer",
        "foldseek_transfer",
        "sequence_similarity_transfer",
        "fold_cluster_transfer",
    ]
    plot_df = df[df["method"].isin(keep)].copy()
    plot_df["method"] = plot_df["method"].map(
        {
            "catalyst_microenvironment": "Catalyst Atlas",
            "catalyst_hybrid": "Hybrid (eng+learned)",
            "esm_gnn_fusion": "ESM-2 + GNN",
            "learned_fusion_encoder": "Learned fusion",
            "learned_catalytic_encoder": "Learned RC encoder",
            "esm2_transfer": "ESM-2 (frozen)",
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

    _plot_fold_disconnected_hero(results)
    if results.get("sequence_identity_stratified"):
        _plot_identity_stratified(results["sequence_identity_stratified"])
    if results.get("fold_chemistry_audits"):
        _plot_fold_chemistry_audits(results["fold_chemistry_audits"])
    if results.get("annotation_style_audits"):
        _plot_annotation_style_audits(results["annotation_style_audits"])


def _plot_fold_disconnected_hero(results: dict[str, Any]) -> None:
    """Hero figure: chemistry transfer when homologous folds are unavailable."""
    fold = results.get("splits", {}).get("fold_cluster", {})
    methods = fold.get("methods") or {}
    order = [
        ("foldseek_transfer", "Foldseek", "#6B7280"),
        ("mmseqs_transfer", "MMseqs2", "#9CA3AF"),
        ("catalyst_microenvironment", "Catalyst Atlas", "#0E7490"),
    ]
    labels, values, colors = [], [], []
    for key, label, color in order:
        if key not in methods:
            continue
        labels.append(label)
        values.append(float(methods[key]["accuracy"]))
        colors.append(color)
    if not labels:
        return

    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    y = list(range(len(labels)))
    ax.barh(y, values, color=colors, height=0.62, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, max(0.55, max(values) * 1.35))
    ax.set_xlabel("Chemistry-family accuracy")
    ax.set_title("Chemistry transfer under fold-disconnected evaluation")
    for yi, v in zip(y, values, strict=True):
        ax.text(v + 0.012, yi, f"{v:.2f}", va="center", ha="left", fontsize=12, color="#1B2A2F")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.text(
        0.02,
        0.02,
        "When homologous folds are unavailable, catalytic microenvironment "
        "representations preserve chemistry information.",
        fontsize=9,
        color="#374151",
        wrap=True,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIGURES / "fig_fold_disconnected_chemistry.png", dpi=180)
    plt.close(fig)


def _plot_identity_stratified(identity_results: dict[str, Any]) -> None:
    """Catalyst vs MMseqs vs Foldseek by nearest-train sequence identity."""
    methods_payload = identity_results.get("methods") or {}
    keep = [
        ("catalyst_microenvironment", "Catalyst Atlas", "#0E7490"),
        ("mmseqs_transfer", "MMseqs2", "#9CA3AF"),
        ("foldseek_transfer", "Foldseek", "#6B7280"),
    ]
    bin_order = [name for name, _, _ in IDENTITY_BINS]
    counts = identity_results.get("bin_counts") or {}
    rows = []
    for key, label, _color in keep:
        if key not in methods_payload:
            continue
        for b in bin_order:
            if b in methods_payload[key]:
                rows.append(
                    {
                        "bin": b,
                        "method": label,
                        "accuracy": float(methods_payload[key][b]),
                        "n": int(counts.get(b, 0)),
                    }
                )
    if not rows:
        return
    plot_df = pd.DataFrame(rows)
    # Tick labels include n so the footer cannot drift from the bars.
    tick_labels = [f"{b}\n(n={counts.get(b, 0)})" for b in bin_order]
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    palette = {
        "Catalyst Atlas": "#0E7490",
        "MMseqs2": "#9CA3AF",
        "Foldseek": "#6B7280",
    }
    sns.barplot(
        data=plot_df,
        x="bin",
        y="accuracy",
        hue="method",
        order=bin_order,
        hue_order=[lab for _, lab, _ in keep if lab in set(plot_df["method"])],
        palette=palette,
        ax=ax,
    )
    ax.set_ylim(0, 1.12)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Nearest train sequence identity")
    ax.set_ylabel("Chemistry-family accuracy")
    ax.set_title("Chemistry transfer vs evolutionary distance")
    ax.legend(title="", loc="upper right", fontsize=9)
    # Value labels on every bar — flat series should still read as numbers.
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)
    src = identity_results.get("identity_source") or ""
    if src:
        fig.text(0.98, 0.01, f"identity: {src}", fontsize=8, color="#6B7280", ha="right")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGURES / "fig_chemistry_by_seq_identity.png", dpi=180)
    plt.close(fig)


def _enrich_meta_for_audits(meta: pd.DataFrame) -> pd.DataFrame:
    """Attach catalytic_aas / cofactor_names / EC coarse labels when missing."""
    from catalyst_atlas.data.uniprot_expand import attach_ec_labels

    out = meta.copy()
    micro_path = PROCESSED / "microenvironments.parquet"
    if micro_path.exists():
        micro = pd.read_parquet(micro_path)
        cols = [
            c
            for c in ("catalytic_aas", "cofactor_names", "first_shell_aas")
            if c in micro.columns
        ]
        if cols and "enzyme_id" in micro.columns:
            m = micro[["enzyme_id", *cols]].drop_duplicates("enzyme_id")
            out["enzyme_id"] = out["enzyme_id"].astype(str)
            m["enzyme_id"] = m["enzyme_id"].astype(str)
            merged = out.merge(m, on="enzyme_id", how="left", suffixes=("", "_micro"))
            for c in cols:
                src = c if c in merged.columns else None
                alt = f"{c}_micro" if f"{c}_micro" in merged.columns else None
                if c not in out.columns and src:
                    out[c] = merged[src]
                elif alt:
                    if c in out.columns:
                        out[c] = out[c].where(out[c].notna(), merged[alt])
                    else:
                        out[c] = merged[alt]
    if "ec_number" in out.columns:
        out = attach_ec_labels(out)
    return out


def _plot_annotation_style_audits(audits: dict[str, Any]) -> None:
    """Bar summary of annotation-style negative controls."""
    ensure_dirs()
    rows = []
    for key, block in audits.items():
        n = int(block.get("n") or 0)
        methods = block.get("methods") or {}
        for mkey, scores in methods.items():
            rows.append(
                {
                    "control": key,
                    "method": mkey,
                    "accuracy": float(scores["accuracy"]),
                    "n": n,
                }
            )
    if not rows:
        return
    df = pd.DataFrame(rows)
    labels = {
        "same_residues_different_chemistry": "Same residues,\ndiff chemistry",
        "same_cofactor_different_chemistry": "Same cofactor,\ndiff chemistry",
        "shuffled_first_shell": "Shuffled\nfirst shell",
        "decoy_reaction_centers": "Decoy\nreaction centers",
    }
    df["control_label"] = df["control"].map(labels).fillna(df["control"])
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    sns.barplot(data=df, x="control_label", y="accuracy", hue="method", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("")
    ax.set_ylabel("Chemistry accuracy")
    ax.set_title("Annotation-style negative controls (random-split test)")
    n_note = "  ·  ".join(
        f"{labels.get(k, k).replace(chr(10), ' ')} n={int((audits.get(k) or {}).get('n') or 0)}"
        for k in labels
        if k in audits
    )
    fig.text(0.02, 0.02, n_note, fontsize=8, color="#374151")
    chance = (audits.get("decoy_reaction_centers") or {}).get("chance_majority")
    if chance is not None:
        ax.axhline(float(chance), color="#9CA3AF", ls="--", lw=1, label=f"chance={chance:.2f}")
    ax.legend(title="", loc="upper right", fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIGURES / "fig_annotation_style_controls.png", dpi=180)
    plt.close(fig)


def _plot_fold_chemistry_audits(audits: dict[str, Any]) -> None:
    """Two-panel: same-fold traps vs different-fold chemistry recovery."""
    panels = [
        (
            "same_fold_different_chemistry",
            "Same fold, different chemistry",
            "False-transfer trap",
        ),
        (
            "different_fold_same_chemistry",
            "Different fold, same chemistry",
            "Convergent recovery",
        ),
    ]
    method_order = [
        ("catalyst_microenvironment", "Catalyst", "#0E7490"),
        ("catalyst_hybrid", "Hybrid", "#047857"),
        ("esm_gnn_fusion", "ESM+GNN", "#065F46"),
        ("learned_fusion_encoder", "Fusion", "#B45309"),
        ("learned_catalytic_encoder", "Learned RC", "#D97706"),
        ("esm2_transfer", "ESM-2", "#475569"),
        ("foldseek_transfer", "Foldseek", "#6B7280"),
        ("mmseqs_transfer", "MMseqs2", "#9CA3AF"),
        ("fold_cluster_transfer", "CATH fold", "#D1D5DB"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, (key, title, subtitle) in zip(axes, panels, strict=True):
        block = audits.get(key) or {}
        methods = block.get("methods") or {}
        n = int(block.get("n") or 0)
        labels, values, colors = [], [], []
        for mkey, label, color in method_order:
            if mkey not in methods:
                continue
            labels.append(label)
            values.append(float(methods[mkey]["accuracy"]))
            colors.append(color)
        if not labels:
            ax.set_title(f"{title}\n(n=0)")
            ax.axis("off")
            continue
        y = list(range(len(labels)))
        ax.barh(y, values, color=colors, height=0.62, edgecolor="none")
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Accuracy")
        ax.set_title(f"{title}\n{subtitle} (n={n})")
        for yi, v in zip(y, values, strict=True):
            ax.text(min(v + 0.02, 0.98), yi, f"{v:.2f}", va="center", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(
        "Fold–chemistry audits (random split; convergent is an informative n-limited audit)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_fold_chemistry_audits.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_eval(
    k: int = 5,
    test_size: float = 0.2,
    seed: int = 7,
    run_external: bool = True,
    threads: int = 4,
    label_col: str | None = None,
) -> dict[str, Any]:
    ensure_dirs()
    meta, X_full, X_comp = _load_unscaled_features()
    meta = _enrich_meta_for_audits(meta)
    X_esm, X_learned, X_fusion, X_esm_gnn, X_esm_gnn_rand = _load_optional_learned_embeddings(
        meta
    )
    label_col = label_col or chemistry_label_col(meta)
    if label_col not in meta.columns:
        raise ValueError(f"label_col={label_col!r} not in meta columns")
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
        "optional_embeddings": {
            "esm2": X_esm is not None,
            "learned_catalytic_encoder": X_learned is not None,
            "learned_fusion_encoder": X_fusion is not None,
            "esm_gnn_fusion": X_esm_gnn is not None,
            "esm_gnn_random_graph": X_esm_gnn_rand is not None,
        },
        "splits": {},
    }
    emb_kw = dict(
        X_esm=X_esm,
        X_learned=X_learned,
        X_fusion=X_fusion,
        X_esm_gnn=X_esm_gnn,
        X_esm_gnn_rand=X_esm_gnn_rand,
    )
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
            **emb_kw,
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
            **emb_kw,
        )
        results["cryptic_seq_holdout"] = sub

    # Secondary analyses on the random split (evolutionary distance + wow-tests).
    rand_train, rand_test = splits["random"]
    y_rand, preds_rand, _, _ = _method_predictions(
        meta,
        X_full,
        X_comp,
        rand_train,
        rand_test,
        k=k,
        seq_sim=seq_sim,
        label_col=label_col,
        mmseqs_hits=mmseqs_hits,
        foldseek_hits=foldseek_hits,
        **emb_kw,
    )
    nearest_id, id_source = nearest_train_sequence_identity(
        meta,
        rand_train,
        rand_test,
        mmseqs_hits=mmseqs_hits,
        seq_sim=seq_sim,
    )
    focus_preds = {
        name: preds_rand[name]
        for name in (
            "catalyst_microenvironment",
            "catalyst_hybrid",
            "esm_gnn_fusion",
            "esm_gnn_random_graph",
            "learned_fusion_encoder",
            "learned_catalytic_encoder",
            "esm2_transfer",
            "mmseqs_transfer",
            "foldseek_transfer",
            "fold_cluster_transfer",
            "sequence_similarity_transfer",
        )
        if name in preds_rand
    }
    identity_block = sequence_identity_stratified_transfer(
        y_rand, focus_preds, nearest_id
    )
    identity_block["identity_source"] = id_source
    identity_block["split"] = "random"
    results["sequence_identity_stratified"] = identity_block
    results["fold_chemistry_audits"] = fold_chemistry_audits(
        meta,
        rand_train,
        rand_test,
        focus_preds,
        label_col=label_col,
    )
    y_train_rand = meta.iloc[rand_train][label_col].astype(str).tolist()
    results["annotation_style_audits"] = annotation_style_audits(
        meta,
        rand_train,
        rand_test,
        focus_preds,
        X_full=X_full,
        y_train=y_train_rand,
        y_test=y_rand,
        label_col=label_col,
        k=k,
        seed=seed,
    )
    # Also report annotation controls under fold holdout (harder regime).
    fold_train, fold_test = splits["fold_cluster"]
    y_fold_te, preds_fold, _, _ = _method_predictions(
        meta,
        X_full,
        X_comp,
        fold_train,
        fold_test,
        k=k,
        seq_sim=seq_sim,
        label_col=label_col,
        mmseqs_hits=mmseqs_hits,
        foldseek_hits=foldseek_hits,
        **emb_kw,
    )
    fold_focus = {n: preds_fold[n] for n in focus_preds if n in preds_fold}
    results["annotation_style_audits_fold_cluster"] = annotation_style_audits(
        meta,
        fold_train,
        fold_test,
        fold_focus,
        X_full=X_full,
        y_train=meta.iloc[fold_train][label_col].astype(str).tolist(),
        y_test=y_fold_te,
        label_col=label_col,
        k=k,
        seed=seed,
    )
    logger.info(
        "Identity-stratified bins (%s): %s",
        id_source,
        identity_block.get("bin_counts"),
    )
    logger.info(
        "Annotation-style controls (random): %s",
        {k: v.get("n") for k, v in results["annotation_style_audits"].items()},
    )

    out_path = PROCESSED / "eval_metrics.json"
    out_path.write_text(json.dumps(results, indent=2))
    _plot_results(results)
    logger.info("Wrote %s and figures under %s", out_path, FIGURES)
    return results
