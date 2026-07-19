"""Query → ESM neighbor vs ESM+GNN neighbor figure."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from sklearn.neighbors import NearestNeighbors

from catalyst_atlas.paths import FIGURES, PROCESSED, ensure_dirs

logger = logging.getLogger(__name__)

TEAL = "#0E7490"
INK = "#1B2A2F"
MUTED = "#4B5563"


def _load_emb(stem: str, meta: pd.DataFrame) -> np.ndarray | None:
    path = PROCESSED / f"embedding_{stem}.npy"
    meta_path = PROCESSED / f"embedding_{stem}_meta.parquet"
    if not path.exists():
        return None
    X = np.load(path)
    if meta_path.exists():
        em = pd.read_parquet(meta_path).reset_index(drop=True)
        if not meta["enzyme_id"].astype(str).equals(em["enzyme_id"].astype(str)):
            emap = {str(e): i for i, e in enumerate(em["enzyme_id"])}
            try:
                X = X[[emap[str(e)] for e in meta["enzyme_id"]]]
            except KeyError:
                return None
    if len(X) != len(meta):
        return None
    return X


def _top1(X: np.ndarray, i: int) -> tuple[int, float]:
    nn = NearestNeighbors(n_neighbors=min(3, len(X)), metric="euclidean")
    nn.fit(X)
    dist, idx = nn.kneighbors(X[i : i + 1], n_neighbors=min(3, len(X)))
    for j, d in zip(idx[0].tolist(), dist[0].tolist(), strict=True):
        if j != i:
            return int(j), float(d)
    return int(idx[0][-1]), float(dist[0][-1])


def _chem(row: pd.Series) -> str:
    return str(row.get("chemistry_family") or row.get("chemistry_class") or "?")


def _fold(row: pd.Series) -> str:
    cath = row.get("cath_topology")
    if cath and str(cath) not in {"", "unknown", "nan"}:
        return str(cath)
    return f"fold={row.get('fold_cluster', '?')}"


def find_retrieval_contrast(
    meta: pd.DataFrame,
    X_esm: np.ndarray,
    X_gnn: np.ndarray,
    max_scan: int = 400,
) -> dict[str, Any] | None:
    """Prefer: ESM NN same-fold-ish; ESM+GNN NN different fold, same chemistry."""
    rng = np.random.default_rng(7)
    order = np.arange(len(meta))
    rng.shuffle(order)
    best = None
    best_score = -1.0
    for i in order[:max_scan]:
        q = meta.iloc[int(i)]
        q_chem = _chem(q)
        q_fold = int(q.get("fold_cluster", -1))
        j_e, d_e = _top1(X_esm, int(i))
        j_g, d_g = _top1(X_gnn, int(i))
        ne = meta.iloc[j_e]
        ng = meta.iloc[j_g]
        if _chem(ne) != q_chem or _chem(ng) != q_chem:
            continue
        e_same_fold = int(ne.get("fold_cluster", -2)) == q_fold
        g_diff_fold = int(ng.get("fold_cluster", -2)) != q_fold
        score = (2.0 if e_same_fold else 0.0) + (3.0 if g_diff_fold else 0.0)
        if j_e == j_g:
            score -= 1.0
        if score > best_score:
            best_score = score
            best = {
                "query_idx": int(i),
                "esm_idx": j_e,
                "gnn_idx": j_g,
                "esm_dist": d_e,
                "gnn_dist": d_g,
                "score": score,
            }
        if score >= 5.0:
            break
    return best


def _fallback_pair(meta: pd.DataFrame) -> dict[str, Any] | None:
    ids = meta["enzyme_id"].astype(str)
    q_ids = ["MCSA00176", "MCSA00623"]
    present = [i for i, eid in enumerate(ids) if eid in q_ids]
    if len(present) < 2:
        return None
    return {
        "query_idx": present[0],
        "esm_idx": present[1],
        "gnn_idx": present[1],
        "esm_dist": float("nan"),
        "gnn_dist": float("nan"),
        "score": 0.0,
        "fallback": True,
    }


def _card(
    ax,
    x,
    y,
    w,
    h,
    title,
    rows: list[tuple[str, str]],
    *,
    fc: str,
    ec: str = INK,
) -> None:
    """Labeled key/value card — same visual language as chemistry score cards."""
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.04",
        linewidth=1.5,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.2,
        y + h - 0.32,
        title,
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color=INK,
    )
    ax.plot([x + 0.18, x + w - 0.18], [y + h - 0.55, y + h - 0.55], color="#D1D5DB", lw=0.8)
    for k, (label, value) in enumerate(rows[:4]):
        yy = y + h - 0.95 - 0.42 * k
        ax.text(x + 0.22, yy, label, ha="left", va="top", fontsize=7.8, color=MUTED)
        ax.text(
            x + 0.22,
            yy - 0.22,
            value,
            ha="left",
            va="top",
            fontsize=9.2,
            color=INK,
            fontweight="bold",
        )


def generate_retrieval_figure(*, dpi: int = 180, path: Path | None = None) -> Path:
    ensure_dirs()
    FIGURES.mkdir(parents=True, exist_ok=True)
    meta_path = PROCESSED / "features_full_meta.parquet"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}")
    meta = pd.read_parquet(meta_path).reset_index(drop=True)
    X_esm = _load_emb("esm", meta)
    X_gnn = _load_emb("esm_gnn", meta)
    if X_esm is None or X_gnn is None:
        raise FileNotFoundError("Need embedding_esm.npy and embedding_esm_gnn.npy")

    pick = find_retrieval_contrast(meta, X_esm, X_gnn) or _fallback_pair(meta)
    if pick is None:
        raise RuntimeError("Could not find a retrieval contrast case")

    q = meta.iloc[pick["query_idx"]]
    ne = meta.iloc[pick["esm_idx"]]
    ng = meta.iloc[pick["gnn_idx"]]

    fig, ax = plt.subplots(figsize=(11.2, 6.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Query on top, centered — no overlap with neighbor cards below.
    _card(
        ax,
        3.0,
        4.2,
        4.0,
        2.35,
        f"Query  {q['enzyme_id']}",
        [
            ("Chemistry", _chem(q)),
            ("Fold / CATH", _fold(q)),
            ("Seq cluster", str(q.get("seq_cluster", "?"))),
        ],
        fc="#EEF6FF",
        ec=TEAL,
    )

    dist_e = f"{pick['esm_dist']:.3f}" if pick["esm_dist"] == pick["esm_dist"] else "n/a"
    dist_g = f"{pick['gnn_dist']:.3f}" if pick["gnn_dist"] == pick["gnn_dist"] else "n/a"

    _card(
        ax,
        0.35,
        0.4,
        4.4,
        3.15,
        f"ESM-2 NN  {ne['enzyme_id']}",
        [
            ("Chemistry", _chem(ne)),
            ("Fold / CATH", _fold(ne)),
            ("Distance", dist_e),
            ("Note", "often same neighborhood"),
        ],
        fc="#F8FAFC",
        ec="#64748B",
    )
    _card(
        ax,
        5.25,
        0.4,
        4.4,
        3.15,
        f"ESM+GNN NN  {ng['enzyme_id']}",
        [
            ("Chemistry", _chem(ng)),
            ("Fold / CATH", _fold(ng)),
            ("Distance", dist_g),
            ("Note", "same chemistry, different fold"),
        ],
        fc="#ECFDF5",
        ec=TEAL,
    )

    # Arrows: query → each neighbor (clear vertical drop, no crossing boxes).
    ax.annotate(
        "",
        xy=(2.55, 3.45),
        xytext=(4.2, 4.35),
        arrowprops=dict(arrowstyle="->", color="#64748B", lw=1.6, connectionstyle="arc3,rad=0.08"),
    )
    ax.annotate(
        "",
        xy=(7.45, 3.45),
        xytext=(5.8, 4.35),
        arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.6, connectionstyle="arc3,rad=-0.08"),
    )

    ax.set_title(
        "Fold-holdout retrieval: sequence neighbors vs chemistry-aware neighbors",
        fontsize=12,
        color=INK,
        pad=10,
    )
    out = Path(path) if path else FIGURES / "fig_retrieval_neighbors.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white", pad_inches=0.2)
    plt.close(fig)
    logger.info("Wrote %s (query=%s)", out, q["enzyme_id"])

    side = FIGURES / "fig_retrieval_neighbors.json"
    side.write_text(
        json.dumps(
            {
                "query": str(q["enzyme_id"]),
                "esm_nn": str(ne["enzyme_id"]),
                "esm_gnn_nn": str(ng["enzyme_id"]),
                "query_chemistry": _chem(q),
                "score": pick.get("score"),
                "fallback": bool(pick.get("fallback")),
            },
            indent=2,
        )
    )
    return out
