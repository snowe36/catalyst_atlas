"""Hard-negative / positive pair mining for chemistry metric learning."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def _ec_broad(ec: Any) -> str:
    s = str(ec or "")
    return s.split(".")[0] if s else ""


def mine_indices(
    meta_rows: list[dict[str, Any]],
    rng: np.random.Generator,
    n_pos_per_anchor: int = 2,
    n_neg_per_anchor: int = 4,
) -> list[tuple[int, int, int]]:
    """Return (anchor, positive, hard_negative) triplets over train indices 0..n-1.

    Positives: same chemistry_family (prefer different fold_cluster).
    Hard negatives: same fold different chemistry; else same EC broad different
    mechanism; else random different chemistry.
    """
    n = len(meta_rows)
    by_chem: dict[str, list[int]] = defaultdict(list)
    by_fold: dict[Any, list[int]] = defaultdict(list)
    by_ec: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(meta_rows):
        chem = str(row.get("chemistry_family") or row.get("chemistry_class") or "")
        by_chem[chem].append(i)
        by_fold[row.get("fold_cluster")].append(i)
        by_ec[_ec_broad(row.get("ec_number"))].append(i)

    triplets: list[tuple[int, int, int]] = []
    for i, row in enumerate(meta_rows):
        chem = str(row.get("chemistry_family") or row.get("chemistry_class") or "")
        fold = row.get("fold_cluster")
        mech = str(row.get("mechanistic_pattern") or "")
        ecb = _ec_broad(row.get("ec_number"))

        pos_cands = [j for j in by_chem.get(chem, []) if j != i]
        if not pos_cands:
            continue
        # Prefer different fold
        pref = [j for j in pos_cands if meta_rows[j].get("fold_cluster") != fold]
        pool = pref or pos_cands
        rng.shuffle(pool)
        positives = pool[:n_pos_per_anchor]

        # Hard negatives
        hard: list[int] = []
        for j in by_fold.get(fold, []):
            if j == i:
                continue
            jchem = str(
                meta_rows[j].get("chemistry_family")
                or meta_rows[j].get("chemistry_class")
                or ""
            )
            if jchem != chem:
                hard.append(j)
        if len(hard) < n_neg_per_anchor:
            for j in by_ec.get(ecb, []):
                if j == i:
                    continue
                jmech = str(meta_rows[j].get("mechanistic_pattern") or "")
                jchem = str(
                    meta_rows[j].get("chemistry_family")
                    or meta_rows[j].get("chemistry_class")
                    or ""
                )
                if jchem != chem or (mech and jmech and jmech != mech):
                    hard.append(j)
        hard = list(dict.fromkeys(hard))
        if len(hard) < n_neg_per_anchor:
            others = [
                j
                for j in range(n)
                if j != i
                and str(
                    meta_rows[j].get("chemistry_family")
                    or meta_rows[j].get("chemistry_class")
                    or ""
                )
                != chem
            ]
            rng.shuffle(others)
            hard.extend(others)
        hard = list(dict.fromkeys(hard))[:n_neg_per_anchor]
        if not hard:
            continue
        for p in positives:
            neg = int(hard[rng.integers(0, len(hard))])
            triplets.append((i, p, neg))
    return triplets
