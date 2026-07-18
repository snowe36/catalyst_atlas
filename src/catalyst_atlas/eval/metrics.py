from __future__ import annotations

from collections import defaultdict

import numpy as np


def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    if not y_true:
        return 0.0
    return float(np.mean([a == b for a, b in zip(y_true, y_pred, strict=True)]))


def macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    f1s = []
    for lab in labels:
        tp = sum(t == lab and p == lab for t, p in zip(y_true, y_pred, strict=True))
        fp = sum(t != lab and p == lab for t, p in zip(y_true, y_pred, strict=True))
        fn = sum(t == lab and p != lab for t, p in zip(y_true, y_pred, strict=True))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s)) if f1s else 0.0


def recall_at_k_chemistry(
    neighbor_labels: list[list[str]],
    y_true: list[str],
    k: int = 5,
) -> float:
    hits = 0
    for labs, truth in zip(neighbor_labels, y_true, strict=True):
        if truth in labs[:k]:
            hits += 1
    return hits / max(len(y_true), 1)


def stratified_accuracy(
    y_true: list[str],
    y_pred: list[str],
    strata: list[str],
) -> dict[str, float]:
    buckets: dict[str, list[bool]] = defaultdict(list)
    for t, p, s in zip(y_true, y_pred, strata, strict=True):
        buckets[s].append(t == p)
    return {k: float(np.mean(v)) for k, v in sorted(buckets.items())}
