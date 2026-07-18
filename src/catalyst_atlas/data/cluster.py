"""Sequence and fold clustering for leakage-aware splits on real data."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _kmer_set(seq: str, k: int = 3) -> set[str]:
    seq = (seq or "").upper().replace(" ", "")
    if len(seq) < k:
        return {seq} if seq else set()
    return {seq[i : i + k] for i in range(len(seq) - k + 1)}


def kmer_jaccard(a: str, b: str, k: int = 3) -> float:
    sa, sb = _kmer_set(a, k=k), _kmer_set(b, k=k)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def kmer_containment(a: str, b: str, k: int = 3) -> float:
    """|A∩B| / min(|A|,|B|) — more sensitive homology proxy than Jaccard for proteins."""
    sa, sb = _kmer_set(a, k=k), _kmer_set(b, k=k)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / min(len(sa), len(sb))


def greedy_sequence_clusters(
    sequences: list[str],
    threshold: float = 0.15,
    k: int = 3,
    metric: str = "jaccard",
    max_rep_len: int = 1500,
) -> list[int]:
    """Greedy clustering by k-mer similarity (BLAST-like neighborhood proxy).

    Prefer mid-length sequences as seeds (very long polyproteins otherwise absorb
    everything under containment). Default metric is Jaccard — containment is
    unsafe when one sequence is much longer than the other.
    """
    sim_fn = kmer_containment if metric == "containment" else kmer_jaccard
    n = len(sequences)

    def _seed_key(i: int) -> tuple:
        L = len(sequences[i] or "")
        # Prefer typical enzyme lengths; demote empty / mega sequences.
        if L < 20 or L > max_rep_len:
            return (2, -L)
        return (0, -L)

    order = sorted(range(n), key=_seed_key)
    reps: list[str] = []
    labels = [-1] * n
    for i in order:
        seq = sequences[i] or ""
        # Empty sequences get singleton clusters (no false homology).
        if len(seq) < 20:
            labels[i] = len(reps)
            reps.append(seq or f"__empty_{i}__")
            continue
        assigned = None
        for cid, rep in enumerate(reps):
            if str(rep).startswith("__empty_"):
                continue
            if sim_fn(seq, rep, k=k) >= threshold:
                assigned = cid
                break
        if assigned is None:
            assigned = len(reps)
            reps.append(seq)
        labels[i] = assigned
    logger.info(
        "Sequence clustering: %d sequences → %d clusters (%s≥%.2f, k=%d)",
        n,
        len(reps),
        metric,
        threshold,
        k,
    )
    return labels


def cath_topology_cluster(cath_id: str | None) -> str:
    """Map a CATH domain id (e.g. 3.40.50.1860) to topology-level fold cluster."""
    if not cath_id or cath_id in {"", "None", "null"}:
        return "unknown"
    parts = str(cath_id).split(".")
    if len(parts) >= 3:
        return ".".join(parts[:3])
    return str(cath_id)


def encode_cluster_ids(labels: list[str | int]) -> list[int]:
    """Map arbitrary cluster labels to dense integer ids."""
    mapping: dict[str, int] = {}
    out: list[int] = []
    for lab in labels:
        key = str(lab)
        if key not in mapping:
            mapping[key] = len(mapping)
        out.append(mapping[key])
    return out


def pairwise_kmer_similarity_matrix(
    sequences: list[str], k: int = 3, metric: str = "jaccard"
) -> np.ndarray:
    """Dense k-mer similarity matrix for small atlases (n ≲ 1500)."""
    sets = [_kmer_set(s, k=k) for s in sequences]
    n = len(sets)
    sim = np.zeros((n, n), dtype=float)
    for i in range(n):
        sim[i, i] = 1.0
        for j in range(i + 1, n):
            a, b = sets[i], sets[j]
            if not a or not b:
                s = 0.0
            elif metric == "containment":
                s = len(a & b) / min(len(a), len(b))
            else:
                s = len(a & b) / len(a | b)
            sim[i, j] = sim[j, i] = s
    return sim


def nearest_neighbor_label_transfer(
    sim: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    labels: list[str],
) -> list[str]:
    """Transfer chemistry from the most similar train sequence (BLAST-style)."""
    preds: list[str] = []
    train_idx = np.asarray(train_idx)
    for ti in test_idx:
        row = sim[ti, train_idx]
        if row.size == 0:
            preds.append("__unseen__")
            continue
        best = int(train_idx[int(np.argmax(row))])
        preds.append(labels[best])
    return preds
