"""CLI-facing search: chemistry identification with neighbor evidence."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

from catalyst_atlas.explain.cards import format_chemistry_card, format_cryptic_hero
from catalyst_atlas.models.embed import load_index, transfer_chemistry
from catalyst_atlas.paths import FIGURES, PROCESSED, REPORTS, ensure_dirs

logger = logging.getLogger(__name__)


def search_enzyme(enzyme_id: str, k: int = 5) -> dict[str, Any]:
    index = load_index(composition_only=False)
    matches = index.meta.index[index.meta["enzyme_id"] == enzyme_id].tolist()
    if not matches:
        raise KeyError(f"Unknown enzyme_id: {enzyme_id}")
    return transfer_chemistry(index, int(matches[0]), k=k)


def _chem_label(row) -> str:
    if hasattr(row, "get"):
        return str(row.get("chemistry_family") or row.get("chemistry_class") or "unknown")
    return str(getattr(row, "chemistry_family", None) or getattr(row, "chemistry_class", "unknown"))


def _baseline_from_cluster(index, query_idx: int, cluster_col: str) -> str:
    q = index.meta.iloc[query_idx]
    same = index.meta[
        (index.meta[cluster_col] == q[cluster_col]) & (index.meta["enzyme_id"] != q["enzyme_id"])
    ]
    if same.empty:
        return "unrelated / no confident transfer"
    labels = [_chem_label(r) for _, r in same.iterrows()]
    return Counter(labels).most_common(1)[0][0]


def find_cryptic_hero(k: int = 5) -> dict[str, Any]:
    """Pick a demo case where sequence/fold proxies mislead but microenvironment works."""
    index = load_index(composition_only=False)
    meta = index.meta
    candidates = []
    for i in range(len(meta)):
        card = transfer_chemistry(index, i, k=k)
        pred = card.get("predicted_chemistry_family") or card.get("predicted_chemistry_class")
        true = card.get("true_chemistry_family") or card.get("true_chemistry_class")
        if pred != true:
            continue
        seq_base = _baseline_from_cluster(index, i, "seq_cluster")
        fold_base = _baseline_from_cluster(index, i, "fold_cluster")
        # Prefer cases where sequence neighborhood chemistry differs or is sparse.
        same_seq = meta[
            (meta["seq_cluster"] == meta.iloc[i]["seq_cluster"])
            & (meta["enzyme_id"] != meta.iloc[i]["enzyme_id"])
        ]
        seq_wrong = seq_base != true or same_seq.empty
        fold_wrong = fold_base != true
        # Neighbors should come from different sequence clusters (cryptic).
        neigh_seq = {n["seq_cluster"] for n in card["neighbors"]}
        cryptic_neighbors = meta.iloc[i]["seq_cluster"] not in neigh_seq or len(neigh_seq) > 1
        if seq_wrong and cryptic_neighbors:
            candidates.append(
                {
                    "idx": i,
                    "card": card,
                    "seq_baseline": seq_base if not same_seq.empty else "unrelated / no confident transfer",
                    "fold_baseline": fold_base,
                    "score": (
                        card["confidence"]
                        + (0.5 if seq_wrong else 0.0)
                        + (0.4 if fold_wrong else 0.0)
                        + (0.2 if cryptic_neighbors else 0.0)
                    ),
                }
            )
    if not candidates:
        # Fallback: first correct prediction.
        for i in range(len(meta)):
            card = transfer_chemistry(index, i, k=k)
            pred = card.get("predicted_chemistry_family") or card.get("predicted_chemistry_class")
            true = card.get("true_chemistry_family") or card.get("true_chemistry_class")
            if pred == true:
                return {
                    "card": card,
                    "seq_baseline": _baseline_from_cluster(index, i, "seq_cluster"),
                    "fold_baseline": _baseline_from_cluster(index, i, "fold_cluster"),
                }
        raise RuntimeError("No suitable hero case found; run the pipeline first")
    best = max(candidates, key=lambda c: c["score"])
    return {
        "card": best["card"],
        "seq_baseline": best["seq_baseline"],
        "fold_baseline": best["fold_baseline"],
    }


def write_hero_case(k: int = 5) -> tuple[Path, dict[str, Any]]:
    ensure_dirs()
    hero = find_cryptic_hero(k=k)
    md = format_cryptic_hero(
        hero["card"],
        seq_baseline_chem=hero["seq_baseline"],
        fold_baseline_chem=hero["fold_baseline"],
    )
    out = REPORTS / "hero_cryptic_chemistry.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    (FIGURES / "hero_cryptic_chemistry.md").write_text(md)
    (PROCESSED / "hero_case.json").write_text(
        __import__("json").dumps(
            {
                "enzyme_id": hero["card"]["query_enzyme_id"],
                "predicted": hero["card"].get("predicted_chemistry_family")
                or hero["card"].get("predicted_chemistry_class"),
                "true": hero["card"].get("true_chemistry_family")
                or hero["card"].get("true_chemistry_class"),
                "seq_baseline": hero["seq_baseline"],
                "fold_baseline": hero["fold_baseline"],
            },
            indent=2,
        )
    )
    logger.info("Wrote hero case → %s", out)
    return out, hero


def search_main_logic(enzyme_id: str | None, demo_hero: bool, k: int) -> str:
    if demo_hero:
        path, hero = write_hero_case(k=k)
        return path.read_text() + "\n\n" + format_chemistry_card(hero["card"])
    if not enzyme_id:
        raise ValueError("Provide --enzyme-id or --demo-hero")
    card = search_enzyme(enzyme_id, k=k)
    return format_chemistry_card(card)
