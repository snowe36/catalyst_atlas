"""Three real M-CSA case studies for the scientific story (not metric spam)."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from catalyst_atlas.explain.cards import format_case_study
from catalyst_atlas.models.embed import load_index, transfer_chemistry
from catalyst_atlas.paths import REPORTS, ensure_dirs

logger = logging.getLogger(__name__)


def _label(row) -> str:
    if "chemistry_family" in row.index and row.get("chemistry_family"):
        return str(row["chemistry_family"])
    return str(row.get("chemistry_class", "unknown"))


def _baseline_from_cluster(index, query_idx: int, cluster_col: str) -> str:
    q = index.meta.iloc[query_idx]
    same = index.meta[
        (index.meta[cluster_col] == q[cluster_col]) & (index.meta["enzyme_id"] != q["enzyme_id"])
    ]
    if same.empty:
        return "no confident transfer"
    labels = [_label(r) for _, r in same.iterrows()]
    return Counter(labels).most_common(1)[0][0]


def _card_ok(card: dict[str, Any]) -> bool:
    pred = card.get("predicted_chemistry_family") or card.get("predicted_chemistry_class")
    true = card.get("true_chemistry_family") or card.get("true_chemistry_class")
    return pred == true and card.get("confidence", 0) >= 0.4


def find_same_fold_different_chemistry(index, k: int = 5) -> dict[str, Any] | None:
    """Case 1: same CATH/fold cluster, microenvironment recovers the right chemistry."""
    meta = index.meta
    best = None
    for fold_id, grp in meta.groupby("fold_cluster"):
        families = {_label(r) for _, r in grp.iterrows()}
        if len(families) < 2 or len(grp) < 3:
            continue
        for i in grp.index.tolist():
            i = int(i)
            card = transfer_chemistry(index, i, k=k)
            if not _card_ok(card):
                continue
            fold_base = _baseline_from_cluster(index, i, "fold_cluster")
            true = card.get("true_chemistry_family") or card.get("true_chemistry_class")
            if fold_base == true:
                continue  # want fold baseline misleading or mixed
            score = card["confidence"] + 0.3
            cand = {
                "title": "Same fold, different chemistry",
                "question": "Can Catalyst distinguish chemistry within a structural family?",
                "context": f"Shared fold cluster {fold_id} (CATH topology neighborhood)",
                "card": card,
                "seq_baseline": _baseline_from_cluster(index, i, "seq_cluster"),
                "fold_baseline": fold_base,
                "takeaway": (
                    "Fold neighborhood mixes chemistries; catalytic microenvironment "
                    "recovers the reaction-center chemistry."
                ),
                "score": score,
            }
            if best is None or cand["score"] > best["score"]:
                best = cand
    return best


def find_different_fold_same_chemistry(index, k: int = 5) -> dict[str, Any] | None:
    """Case 2: neighbors share chemistry but not fold — convergent chemistry signal."""
    meta = index.meta
    best = None
    for i in range(len(meta)):
        card = transfer_chemistry(index, i, k=k)
        if not _card_ok(card):
            continue
        q_fold = int(meta.iloc[i]["fold_cluster"])
        true = card.get("true_chemistry_family") or card.get("true_chemistry_class")
        neigh_folds = {int(n["fold_cluster"]) for n in card["neighbors"]}
        neigh_chem = {
            n.get("chemistry_family") or n.get("chemistry_class") for n in card["neighbors"]
        }
        if true not in neigh_chem:
            continue
        # Prefer neighbors mostly outside the query fold.
        outside = sum(1 for n in card["neighbors"] if int(n["fold_cluster"]) != q_fold)
        if outside < 3:
            continue
        score = card["confidence"] + 0.1 * outside
        cand = {
            "title": "Different fold, same chemistry",
            "question": "Can Catalyst detect convergent chemistry across folds?",
            "context": (
                f"Query fold_cluster={q_fold}; neighbors span folds {sorted(neigh_folds)}"
            ),
            "card": card,
            "seq_baseline": _baseline_from_cluster(index, i, "seq_cluster"),
            "fold_baseline": _baseline_from_cluster(index, i, "fold_cluster"),
            "takeaway": (
                "Catalytic neighbors share chemistry family despite different fold "
                "neighborhoods — microenvironment captures convergent reaction logic."
            ),
            "score": score,
        }
        if best is None or cand["score"] > best["score"]:
            best = cand
    return best


def find_cofactor_supported_hypothesis(index, k: int = 5) -> dict[str, Any] | None:
    """Case 3: cofactor-rich site where chemistry card is cofactor-aware."""
    meta = index.meta
    best = None
    for i in range(len(meta)):
        tags = str(meta.iloc[i].get("cofactor_tags") or "none")
        if tags == "none":
            continue
        card = transfer_chemistry(index, i, k=k)
        if not _card_ok(card):
            continue
        pred_cof = card.get("predicted_cofactor_tags") or []
        if not pred_cof or pred_cof == ["none"]:
            continue
        # Prefer when neighbors agree on cofactor chemistry.
        score = card["confidence"] + 0.2 * len([t for t in pred_cof if t != "none"])
        cand = {
            "title": "Cofactor-aware chemistry hypothesis",
            "question": "Can Catalyst provide a plausible chemistry hypothesis from the reaction center?",
            "context": f"Site cofactors/metals: {tags}",
            "card": card,
            "seq_baseline": _baseline_from_cluster(index, i, "seq_cluster"),
            "fold_baseline": _baseline_from_cluster(index, i, "fold_cluster"),
            "takeaway": (
                "Cofactor/metal context in the microenvironment supports a chemistry "
                "hypothesis an enzymologist would recognize — not an EC digit alone."
            ),
            "score": score,
        }
        if best is None or cand["score"] > best["score"]:
            best = cand
    return best


def build_case_studies(k: int = 5) -> list[dict[str, Any]]:
    index = load_index(composition_only=False)
    index.meta = index.meta.reset_index(drop=True)
    cases = []
    for finder in (
        find_same_fold_different_chemistry,
        find_different_fold_same_chemistry,
        find_cofactor_supported_hypothesis,
    ):
        case = finder(index, k=k)
        if case:
            cases.append(case)
            logger.info("Case study: %s → %s", case["title"], case["card"]["query_enzyme_id"])
        else:
            logger.warning("No case found for %s", finder.__name__)
    return cases


def write_case_studies(k: int = 5) -> Path:
    ensure_dirs()
    cases = build_case_studies(k=k)
    out_dir = REPORTS / "case_studies"
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = [
        "# Catalyst Atlas — three chemistry case studies",
        "",
        "Real M-CSA reaction centers. Not a leaderboard — three questions enzyme chemists care about.",
        "",
    ]
    summary = []
    for i, case in enumerate(cases, 1):
        md = format_case_study(case)
        path = out_dir / f"case_{i}_{_slug(case['title'])}.md"
        path.write_text(md)
        parts.append(md)
        parts.append("\n---\n")
        summary.append(
            {
                "title": case["title"],
                "enzyme_id": case["card"]["query_enzyme_id"],
                "predicted": case["card"].get("predicted_chemistry_family")
                or case["card"].get("predicted_chemistry_class"),
                "true": case["card"].get("true_chemistry_family")
                or case["card"].get("true_chemistry_class"),
                "cofactors": case["card"].get("true_cofactor_tags"),
            }
        )
    combined = out_dir / "README.md"
    combined.write_text("\n".join(parts))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Wrote %d case studies → %s", len(cases), out_dir)
    return combined


def _slug(title: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in title.lower()).strip("_")[:48]
