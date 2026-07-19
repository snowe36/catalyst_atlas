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
    if not wrong:
        return []
    a = composition[anchor]
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
    max_per_family: int = 4,
) -> list[int]:
    """Sample a class-balanced batch that forces convergent co-occurrence.

    For each sampled chemistry family that spans ≥2 fold clusters, the batch
    includes members from at least two distinct folds (required, not preferred).
    Caps per-family count so the batch retains many negatives for SupCon.
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

    eligible = [c for c, idxs in by_chem.items() if len(idxs) >= min_per_family]
    if not eligible:
        n = len(meta_rows)
        return rng.choice(n, size=min(batch_size, n), replace=False).tolist()

    # Keep ≥4 chemistries in-batch when available so SupCon has real negatives.
    n_fam_target = min(len(eligible), max(4, batch_size // max_per_family))
    fam_cap = max(min_per_family, int(np.ceil(batch_size / n_fam_target)))
    fam_cap = min(fam_cap, max_per_family)

    selected: list[int] = []
    seen: set[int] = set()
    per_chem: dict[str, int] = defaultdict(int)

    def _add(idx: int, chem: str | None = None, cap: int | None = None) -> bool:
        if idx in seen or len(selected) >= batch_size:
            return False
        c = chem if chem is not None else _chem(meta_rows[idx])
        limit = fam_cap if cap is None else cap
        if per_chem[c] >= limit:
            return False
        selected.append(idx)
        seen.add(idx)
        per_chem[c] += 1
        return True

    families = list(eligible)
    rng.shuffle(families)
    families = families[:n_fam_target]

    for chem in families:
        if len(selected) >= batch_size:
            break
        fold_map = by_chem_fold[chem]
        folds = [f for f, idxs in fold_map.items() if idxs]
        taken = 0
        if len(folds) >= 2:
            rng.shuffle(folds)
            for f in folds[:2]:
                pick = int(rng.choice(fold_map[f]))
                if _add(pick, chem):
                    taken += 1
        rest = [i for i in by_chem[chem] if i not in seen]
        rng.shuffle(rest)
        for i in rest:
            if taken >= fam_cap or len(selected) >= batch_size:
                break
            if _add(i, chem):
                taken += 1

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
                # Soft cap +1 so hard negs can enter without collapsing diversity.
                if _add(j, cap=fam_cap + 1):
                    pass

    # Top up, gradually relaxing the per-family cap if needed.
    remaining = [i for i in range(len(meta_rows)) if i not in seen]
    rng.shuffle(remaining)
    relax = fam_cap
    while len(selected) < batch_size and remaining:
        progress = False
        for i in remaining:
            if _add(i, cap=relax):
                progress = True
            if len(selected) >= batch_size:
                break
        if not progress:
            relax += 1
            if relax > batch_size:
                break
        remaining = [i for i in remaining if i not in seen]

    rng.shuffle(selected)
    return selected[:batch_size]
