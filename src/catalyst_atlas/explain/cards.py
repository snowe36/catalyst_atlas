"""Render retrieval-augmented chemistry cards."""

from __future__ import annotations

from typing import Any


def format_chemistry_card(card: dict[str, Any], show_truth: bool = True) -> str:
    lines = [
        "╔══════════════════════════════════════════════════════════╗",
        "║                 CATALYST ATLAS — CHEMISTRY CARD          ║",
        "╚══════════════════════════════════════════════════════════╝",
        f"Query:              {card['query_enzyme_id']}",
        f"Predicted chemistry:{card['predicted_chemistry_class']}",
        f"Catalytic pattern:  {card['predicted_catalytic_pattern']}",
        f"Cofactor / metal:   {', '.join(card['predicted_cofactor_tags'])}",
        f"Confidence:         {card['confidence']:.2f}",
    ]
    if show_truth and card.get("true_chemistry_class") is not None:
        lines += [
            "── ground truth (eval / demo) ──",
            f"True chemistry:     {card['true_chemistry_class']}",
            f"True pattern:       {card['true_catalytic_pattern']}",
            f"True cofactors:     {card['true_cofactor_tags']}",
        ]
    lines.append("── catalytic neighbors (evidence) ──")
    for i, n in enumerate(card.get("neighbors", []), 1):
        lines.append(
            f"  {i}. {n['enzyme_id']}  chem={n['chemistry_class']}  "
            f"pattern={n['catalytic_pattern']}  d={n['distance']:.3f}  "
            f"seq_cluster={n['seq_cluster']} fold_cluster={n['fold_cluster']}"
        )
    return "\n".join(lines)


def format_cryptic_hero(
    card: dict[str, Any],
    seq_baseline_chem: str,
    fold_baseline_chem: str,
    seq_identity_note: str = "~18–22% to nearest labeled neighbor (demo)",
) -> str:
    correct = card["predicted_chemistry_class"] == card.get("true_chemistry_class")
    lines = [
        "# Cryptic chemistry case",
        "",
        f"**Query enzyme:** `{card['query_enzyme_id']}`",
        f"**Sequence identity context:** {seq_identity_note}",
        "",
        "| Method | Inferred chemistry |",
        "|---|---|",
        f"| BLAST / sequence-cluster proxy | `{seq_baseline_chem}` |",
        f"| Foldseek / fold-cluster proxy | `{fold_baseline_chem}` |",
        f"| **Catalyst Atlas** | `{card['predicted_chemistry_class']}` |",
        "",
        "### Chemistry card",
        f"- **Reaction chemistry:** {card['predicted_chemistry_class']}",
        f"- **Catalytic pattern:** {card['predicted_catalytic_pattern']}",
        f"- **Cofactor / metal:** {', '.join(card['predicted_cofactor_tags'])}",
        f"- **Confidence:** {card['confidence']:.2f}",
        "",
        "### Evidence (top catalytic neighbors)",
    ]
    for n in card.get("neighbors", [])[:5]:
        lines.append(
            f"- `{n['enzyme_id']}` — {n['chemistry_class']} / {n['catalytic_pattern']} "
            f"(distance {n['distance']:.3f}; seq_cluster={n['seq_cluster']})"
        )
    if card.get("true_chemistry_class"):
        lines += [
            "",
            f"**Ground truth:** {card['true_chemistry_class']} / {card['true_catalytic_pattern']}",
            f"**Catalyst Atlas correct:** {'yes' if correct else 'no'}",
        ]
    lines += [
        "",
        "> Representation is the **catalytic microenvironment** "
        "(chemistry residues, cofactors, catalytic geometry, ligand contacts) — "
        "not whole-protein fold similarity or pocket shape alone.",
    ]
    return "\n".join(lines)
