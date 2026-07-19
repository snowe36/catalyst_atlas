"""Class-balanced batch sampling for chemistry contrastive learning.

Forces different-fold / same-chemistry co-occurrence and composition-aware
hard negatives so the encoder must learn convergent chemistry signal.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def _chem(row: dict[str, Any]) -> str:
    return str(row.get("chemistry_family") or row.get("chemistry_class") or "")


def _fold(row: dict[str, Any]) -> Any:
    return row.get("fold_cluster")


def mine_indices(
    meta_rows: list[dict[str, Any]],
    rng: np.random.Generator,
    n_pos_per_anchor: int = 2,
    n_neg_per_anchor: int = 4,
) -> list[tuple[int, int, int]]:
    """Legacy triplet miner (kept for tests / fallback).

    Positives: same chemistry_family (prefer different fold_cluster).
    Hard negatives: same fold different chemistry; else random different chemistry.
    """
    n = len(meta_rows)
    by_chem: dict[str, list[int]] = defaultdict(list)
    by_fold: dict[Any, list[int]] = defaultdict(list)
    for i, row in enumerate(meta_rows):
        by_chem[_chem(row)].append(i)
        by_fold[_fold(row)].append(i)

    triplets: list[tuple[int, int, int]] = []
    for i, row in enumerate(meta_rows):
        chem = _chem(row)
        fold = _fold(row)
        pos_cands = [j for j in by_chem.get(chem, []) if j != i]
        if not pos_cands:
            continue
        pref = [j for j in pos_cands if _fold(meta_rows[j]) != fold]
        pool = pref or pos_cands
        rng.shuffle(pool)
        positives = pool[:n_pos_per_anchor]

        hard: list[int] = []
        for j in by_fold.get(fold, []):
            if j != i and _chem(meta_rows[j]) != chem:
                hard.append(j)
        if len(hard) < n_neg_per_anchor:
            others = [j for j in range(n) if j != i and _chem(meta_rows[j]) != chem]
            rng.shuffle(others)
            hard.extend(others)
        hard = list(dict.fromkeys(hard))[:n_neg_per_anchor]
        if not hard:
            continue
        for p in positives:
            neg = int(hard[rng.integers(0, len(hard))])
            triplets.append((i, p, neg))
    return triplets


def _composition_hard_negatives(
    anchor: int,
    meta_rows: list[dict[str, Any]],
    composition: np.ndarray,
    rng: np.random.Generator,
    n: int = 2,
    pool_size: int = 64,
) -> list[int]:
    """Nearest-composition enzymes with different chemistry."""
    chem = _chem(meta_rows[anchor])
    wrong = [j for j in range(len(meta_rows)) if j != anchor and _chem(meta_rows[j]) != chem]
    if not wrong or composition is None:
        return []
    a = composition[anchor]
    # Sample a pool then rank by L2 composition distance (cheap for n≈1k).
    rng.shuffle(wrong)
    pool = wrong[: max(pool_size, n)]
    dists = np.linalg.norm(composition[pool] - a[None, :], axis=1)
    order = np.argsort(dists)
    return [int(pool[i]) for i in order[:n]]


def sample_contrastive_batch(
    meta_rows: list[dict[str, Any]],
    rng: np.random.Generator,
    batch_size: int = 32,
    composition: np.ndarray | None = None,
    n_comp_hard: int = 2,
    min_per_family: int = 2,
) -> list[int]:
    """Sample a class-balanced batch that forces convergent co-occurrence.

    For each sampled chemistry family that spans ≥2 fold clusters, the batch
    includes members from at least two distinct folds (required, not preferred).
    Also injects composition-similar wrong-chemistry hard negatives.
    """
    by_chem: dict[str, list[int]] = defaultdict(list)
    by_chem_fold: dict[str, dict[Any, list[int]]] = defaultdict(lambda: defaultdict(list))
    for i, row in enumerate(meta_rows):
        c = _chem(row)
        if not c:
            continue
        by_chem[c].append(i)
        by_chem_fold[c][_fold(row)].append(i)

    # Families with enough members to form positives.
    eligible = [c for c, idxs in by_chem.items() if len(idxs) >= min_per_family]
    if not eligible:
        # Degenerate: just sample randomly.
        n = len(meta_rows)
        return rng.choice(n, size=min(batch_size, n), replace=False).tolist()

    selected: list[int] = []
    seen: set[int] = set()
    rng.shuffle(eligible)

    def _add(idx: int) -> None:
        if idx not in seen and len(selected) < batch_size:
            selected.append(idx)
            seen.add(idx)

    # Fill with chemistry families, forcing multi-fold when available.
    for chem in eligible:
        if len(selected) >= batch_size:
            break
        fold_map = by_chem_fold[chem]
        folds = [f for f, idxs in fold_map.items() if idxs]
        if len(folds) >= 2:
            rng.shuffle(folds)
            # Require ≥2 distinct folds.
            for f in folds[:2]:
                pick = int(rng.choice(fold_map[f]))
                _add(pick)
            # Extra members from remaining folds / same folds.
            rest = [i for i in by_chem[chem] if i not in seen]
            rng.shuffle(rest)
            for i in rest:
                if len(selected) >= batch_size:
                    break
                _add(i)
        else:
            rest = list(by_chem[chem])
            rng.shuffle(rest)
            for i in rest[:min_per_family]:
                _add(i)

    # Composition-aware hard negatives (wrong chemistry, similar catalytic AA).
    if composition is not None and selected and n_comp_hard > 0:
        anchors = list(selected)
        rng.shuffle(anchors)
        for a in anchors[: max(1, len(anchors) // 2)]:
            if len(selected) >= batch_size:
                break
            for j in _composition_hard_negatives(
                a, meta_rows, composition, rng, n=n_comp_hard
            ):
                _add(j)
                if len(selected) >= batch_size:
                    break

    # Top up with random indices if under-filled.
    if len(selected) < batch_size:
        remaining = [i for i in range(len(meta_rows)) if i not in seen]
        rng.shuffle(remaining)
        for i in remaining:
            _add(i)
            if len(selected) >= batch_size:
                break

    rng.shuffle(selected)
    return selected
