"""Render retrieval-augmented chemistry cards."""

from __future__ import annotations

from typing import Any


def _pred(card: dict[str, Any]) -> str:
    return str(
        card.get("predicted_chemistry_family") or card.get("predicted_chemistry_class") or "unknown"
    )


def _true(card: dict[str, Any]) -> str | None:
    val = card.get("true_chemistry_family") or card.get("true_chemistry_class")
    return str(val) if val is not None else None


def build_catalytic_evidence(card: dict[str, Any]) -> list[str]:
    """Human-readable mechanistic evidence lines for the product card."""
    evidence: list[str] = []
    mech = str(card.get("predicted_mechanistic_pattern") or card.get("true_mechanistic_pattern") or "")
    pattern = str(card.get("predicted_catalytic_pattern") or card.get("true_catalytic_pattern") or "")
    cofs = card.get("predicted_cofactor_tags") or []
    if isinstance(cofs, str):
        cofs = [c.strip() for c in cofs.split(",") if c.strip()]
    true_cofs = str(card.get("true_cofactor_tags") or "none")
    query_cofs = [t.strip() for t in true_cofs.split(",") if t.strip() and t.strip() != "none"]

    if pattern and pattern != "unknown":
        evidence.append(f"catalytic residue pattern: {pattern}")
    if mech and mech != "unknown":
        evidence.append(f"mechanistic pattern: {mech}")
    for tag in query_cofs:
        evidence.append(f"{tag} cofactor/metal detected at reaction center")
    # Coordination motifs from query site (if attached to card)
    for coord in card.get("metal_coordination") or []:
        motif = coord.get("motif") or ""
        geom = coord.get("geometry") or ""
        metal = coord.get("metal") or "metal"
        if motif:
            evidence.append(f"{metal} coordination: {motif} ({geom})")
        elif geom:
            evidence.append(f"{metal} geometry: {geom}")
    if card.get("confidence", 0) >= 0.6:
        evidence.append("neighbor consensus supports chemistry family")
    # Convergent / distant analogs
    folds = {n.get("fold_cluster") for n in card.get("neighbors") or []}
    if len(folds) > 1:
        evidence.append("chemical analogs span multiple fold neighborhoods")
    if not evidence:
        evidence.append("shared catalytic microenvironment with nearest neighbors")
    return evidence[:6]


def format_product_card(card: dict[str, Any], show_truth: bool = False) -> str:
    """Portfolio / CLI product card — mechanistically grounded, not a score dump."""
    pred = _pred(card)
    mech = card.get("predicted_mechanistic_pattern") or "unknown"
    conf = float(card.get("confidence") or 0.0)
    evidence = build_catalytic_evidence(card)

    lines = [
        "Catalyst Atlas prediction",
        "=========================",
        "",
        "Chemistry:",
        f"  {pred}",
        f"  ({mech})",
        "",
        "Confidence:",
        f"  {conf:.2f}",
        "",
        "Catalytic evidence:",
    ]
    for e in evidence:
        lines.append(f"  ✓ {e}")

    lines += ["", "Closest chemical analogs:"]
    for i, n in enumerate(card.get("neighbors") or [], 1):
        chem = n.get("chemistry_family") or n.get("chemistry_class") or "?"
        cof = n.get("cofactor_tags") or "none"
        q_fold = card.get("query_fold_cluster")
        if q_fold is not None and n.get("fold_cluster") != q_fold:
            note = "different fold"
        else:
            note = "shared fold neighborhood"
        lines.append(
            f"  {i}. {n['enzyme_id']} — {chem} / {n.get('mechanistic_pattern', '?')} "
            f"(cof={cof}; {note}; d={n['distance']:.2f})"
        )

    lines += [
        "",
        "Why:",
        "  Shared catalytic microenvironment",
        "  (reaction-center residues + cofactor/metal geometry — not fold TM-score)",
    ]

    if show_truth and _true(card) is not None:
        lines += [
            "",
            f"[eval] ground truth: {_true(card)} / {card.get('true_mechanistic_pattern', '—')}",
            f"[eval] correct: {'yes' if _pred(card) == _true(card) else 'no'}",
        ]
    return "\n".join(lines)


def format_chemistry_card(card: dict[str, Any], show_truth: bool = True) -> str:
    """Default CLI card = product card; keep a compact legacy block below if useful."""
    return format_product_card(card, show_truth=show_truth)


def format_cryptic_hero(
    card: dict[str, Any],
    seq_baseline_chem: str,
    fold_baseline_chem: str,
    seq_identity_note: str = "sequence / fold retrieval baselines vs microenvironment",
    title: str = "Cryptic chemistry case",
) -> str:
    pred = _pred(card)
    true = _true(card)
    correct = pred == true
    lines = [
        f"# {title}",
        "",
        f"**Query enzyme:** `{card['query_enzyme_id']}`"
        + (f" — {card['query_enzyme_name']}" if card.get("query_enzyme_name") else ""),
        f"**Context:** {seq_identity_note}",
        "",
        "| Method | Inferred chemistry |",
        "|---|---|",
        f"| Sequence retrieval baseline | `{seq_baseline_chem}` |",
        f"| Fold / CATH retrieval baseline | `{fold_baseline_chem}` |",
        f"| **Catalyst Atlas** | `{pred}` |",
        "",
        "```",
        format_product_card(card, show_truth=False),
        "```",
    ]
    if true:
        lines += [
            "",
            f"**Ground truth:** {true} / {card.get('true_mechanistic_pattern', '—')}",
            f"**Catalyst Atlas correct:** {'yes' if correct else 'no'}",
        ]
    lines += [
        "",
        "> Representation is the **catalytic microenvironment** "
        "(reaction-center residues, cofactors/metals, geometry, first shell) — "
        "not whole-protein fold similarity or pocket shape alone.",
    ]
    return "\n".join(lines)


def format_case_study(case: dict[str, Any]) -> str:
    """Render one of the three scientific case studies."""
    card = case["card"]
    lines = [
        f"# Case study: {case['title']}",
        "",
        f"**Question:** {case['question']}",
        "",
        format_cryptic_hero(
            card,
            seq_baseline_chem=case.get("seq_baseline", "—"),
            fold_baseline_chem=case.get("fold_baseline", "—"),
            seq_identity_note=case.get("context", ""),
            title=case["title"],
        ),
    ]
    if case.get("takeaway"):
        lines += ["", f"**Takeaway:** {case['takeaway']}"]
    return "\n".join(lines)
