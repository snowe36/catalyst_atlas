"""Query → ESM neighbor vs ESM+GNN neighbor figure."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from catalyst_atlas.paths import FIGURES, PROCESSED, ensure_dirs

logger = logging.getLogger(__name__)


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
        # Score: want ESM same-fold + GNN cross-fold same chemistry.
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
    """Thermolysin / neprilysin if present."""
    ids = meta["enzyme_id"].astype(str)
    q_ids = ["MCSA00176", "MCSA00623"]
    present = [i for i, eid in enumerate(ids) if eid in q_ids]
    if len(present) < 2:
        return None
    # query first, neighbor second
    qi = present[0]
    ni = present[1]
    return {
        "query_idx": qi,
        "esm_idx": ni,
        "gnn_idx": ni,
        "esm_dist": float("nan"),
        "gnn_dist": float("nan"),
        "score": 0.0,
        "fallback": True,
    }


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

    pick = find_retrieval_contrast(meta, X_esm, X_gnn)
    if pick is None:
        pick = _fallback_pair(meta)
    if pick is None:
        raise RuntimeError("Could not find a retrieval contrast case")

    q = meta.iloc[pick["query_idx"]]
    ne = meta.iloc[pick["esm_idx"]]
    ng = meta.iloc[pick["gnn_idx"]]

    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def box(x, y, w, h, title, lines, fc="#F4F7FA"):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.15",
            linewidth=1.2,
            edgecolor="#334155",
            facecolor=fc,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h - 0.35, title, ha="center", va="top", fontsize=11, fontweight="bold")
        for k, line in enumerate(lines):
            ax.text(x + 0.2, y + h - 0.85 - 0.45 * k, line, ha="left", va="top", fontsize=9.5)

    box(
        3.5,
        1.7,
        3.0,
        2.4,
        f"Query  {q['enzyme_id']}",
        [_chem(q), _fold(q), f"seq_cluster={q.get('seq_cluster', '?')}"],
        fc="#EEF6FF",
    )
    box(
        0.4,
        0.4,
        3.2,
        2.6,
        f"ESM-2 NN  {ne['enzyme_id']}",
        [
            _chem(ne),
            _fold(ne),
            f"dist={pick['esm_dist']:.3f}" if pick["esm_dist"] == pick["esm_dist"] else "dist=n/a",
            "often same neighborhood",
        ],
        fc="#F8FAFC",
    )
    box(
        6.4,
        0.4,
        3.2,
        2.6,
        f"ESM+GNN NN  {ng['enzyme_id']}",
        [
            _chem(ng),
            _fold(ng),
            f"dist={pick['gnn_dist']:.3f}" if pick["gnn_dist"] == pick["gnn_dist"] else "dist=n/a",
            "same chemistry, different fold",
        ],
        fc="#ECFDF5",
    )
    ax.annotate(
        "",
        xy=(3.6, 2.2),
        xytext=(3.5, 2.8),
        arrowprops=dict(arrowstyle="->", color="#64748B", lw=1.4),
    )
    ax.annotate(
        "",
        xy=(6.4, 2.2),
        xytext=(6.5, 2.8),
        arrowprops=dict(arrowstyle="->", color="#64748B", lw=1.4),
    )
    ax.set_title(
        "Fold-holdout retrieval: sequence neighbors vs chemistry-aware neighbors",
        fontsize=12,
        pad=8,
    )
    out = Path(path) if path else FIGURES / "fig_retrieval_neighbors.png"
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s (query=%s)", out, q["enzyme_id"])
    # Sidecar for README / case studies.
    side = FIGURES / "fig_retrieval_neighbors.json"
    side.write_text(
        __import__("json").dumps(
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
