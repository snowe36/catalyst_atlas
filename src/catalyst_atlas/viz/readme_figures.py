"""Pipeline and chemistry-card figures for the README."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch

from catalyst_atlas.paths import FIGURES, ensure_dirs

logger = logging.getLogger(__name__)

TEAL = "#0E7490"
INK = "#1B2A2F"
MUTED = "#4B5563"
RULE = "#D1D5DB"
CARD_BG = "#F8FAFB"


def render_pipeline_figure(path: Path | None = None, *, dpi: int = 180) -> Path:
    """Figure 1 — structure → reaction center → representation → retrieval → card."""
    ensure_dirs()
    out = Path(path) if path else FIGURES / "fig1_pipeline.png"
    steps = [
        ("Structure", "M-CSA site + PDB"),
        ("Reaction center", "residues · geometry\ncofactors · first shell"),
        ("Representation", "engineered features"),
        ("Retrieval", "catalytic neighbors"),
        ("Chemistry card", "family · pattern\nevidence"),
    ]
    fig, ax = plt.subplots(figsize=(11.2, 2.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    n = len(steps)
    box_w, box_h = 0.155, 0.58
    gap = (1.0 - n * box_w) / (n + 1)
    y0 = 0.22
    for i, (title, sub) in enumerate(steps):
        x0 = gap + i * (box_w + gap)
        color = TEAL if i == n - 1 else INK
        face = "#E6F4F7" if i == n - 1 else "#F3F4F6"
        box = FancyBboxPatch(
            (x0, y0),
            box_w,
            box_h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.4,
            edgecolor=color,
            facecolor=face,
        )
        ax.add_patch(box)
        ax.text(
            x0 + box_w / 2,
            y0 + box_h * 0.68,
            title,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=color,
            linespacing=1.15,
        )
        ax.text(
            x0 + box_w / 2,
            y0 + box_h * 0.28,
            sub,
            ha="center",
            va="center",
            fontsize=8,
            color="#374151",
            linespacing=1.25,
        )
        if i < n - 1:
            ax.annotate(
                "",
                xy=(x0 + box_w + gap * 0.55, y0 + box_h / 2),
                xytext=(x0 + box_w + gap * 0.1, y0 + box_h / 2),
                arrowprops=dict(arrowstyle="->", color="#6B7280", lw=1.6),
            )
    ax.set_title(
        "Structure → reaction center → chemistry",
        fontsize=12,
        color=INK,
        pad=8,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Wrote pipeline figure → %s", out)
    return out


def _draw_score_card(ax, card: dict) -> None:
    """One chemistry evidence card with stable vertical bands (no overflow)."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    frame = FancyBboxPatch(
        (0.02, 0.03),
        0.96,
        0.94,
        boxstyle="round,pad=0.018,rounding_size=0.03",
        linewidth=1.6,
        edgecolor=TEAL,
        facecolor=CARD_BG,
    )
    ax.add_patch(frame)

    # Header band
    ax.text(0.07, 0.90, card["title"], fontsize=11, fontweight="bold", color=TEAL, va="center")
    ax.text(0.07, 0.82, card["query"], fontsize=10, fontweight="bold", color=INK, va="center")
    ax.text(0.07, 0.755, card["context"], fontsize=8.2, color=MUTED, va="center")

    ax.plot([0.07, 0.93], [0.70, 0.70], color=RULE, lw=0.9)

    # Score / prediction band
    ax.text(0.07, 0.64, "Predicted chemistry", fontsize=8, color=MUTED, va="center")
    ax.text(0.07, 0.57, card["prediction"], fontsize=12, fontweight="bold", color=TEAL, va="center")
    conf = card.get("confidence")
    if conf is not None:
        badge = FancyBboxPatch(
            (0.72, 0.545),
            0.21,
            0.08,
            boxstyle="round,pad=0.008,rounding_size=0.02",
            linewidth=0,
            facecolor="#DCF5F9",
        )
        ax.add_patch(badge)
        ax.text(
            0.825,
            0.585,
            f"conf {conf:.2f}",
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color=TEAL,
        )

    ax.text(
        0.07, 0.49, f"Top analog · {card['analog']}", fontsize=8.2, color="#374151", va="center"
    )

    ax.plot([0.07, 0.93], [0.44, 0.44], color=RULE, lw=0.9)

    # Evidence band — fixed slots so bullets never collide with the frame
    ax.text(0.07, 0.38, "Evidence", fontsize=9, fontweight="bold", color=INK, va="center")
    why = list(card.get("why") or [])[:4]
    y0 = 0.31
    dy = 0.065
    for i, line in enumerate(why):
        y = y0 - i * dy
        ax.text(0.08, y, "✓", fontsize=9, color=TEAL, va="center", fontweight="bold")
        ax.text(0.13, y, line, fontsize=8.4, color=INK, va="center")


def render_chemistry_cards_figure(path: Path | None = None, *, dpi: int = 180) -> Path:
    """Figure 4 — example chemistry score cards."""
    ensure_dirs()
    out = Path(path) if path else FIGURES / "fig4_chemistry_cards.png"

    cards = [
        {
            "title": "Convergent chemistry",
            "query": "MCSA00176 · thermolysin",
            "context": "Remote sequence · distinct fold · shared Zn hydrolysis",
            "prediction": "hydrolysis / metal activation",
            "confidence": 0.82,
            "analog": "MCSA00623 · neprilysin (different fold)",
            "why": [
                "Zn metal at reaction center",
                "metal-activation pattern",
                "His/Asp/Glu catalytic arrangement",
                "analogs span multiple folds",
            ],
        },
        {
            "title": "Same fold, different chemistry",
            "query": "MCSA00034 · catechol 2,3-dioxygenase",
            "context": "Shared fold neighborhood mixes chemistries",
            "prediction": "oxidation-reduction / metal activation",
            "confidence": 0.74,
            "analog": "Fe-redox catalytic neighbors",
            "why": [
                "Fe cofactor at reaction center",
                "His-Asp catalytic pattern",
                "neighbors disagree with fold prior",
                "microenvironment > fold lookup",
            ],
        },
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.4))
    for ax, card in zip(axes, cards, strict=True):
        _draw_score_card(ax, card)

    fig.suptitle("Example chemistry cards", fontsize=12, color=INK, y=0.995)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.04, wspace=0.06)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white", pad_inches=0.15)
    plt.close(fig)
    logger.info("Wrote chemistry-card figure → %s", out)
    return out


def generate_readme_figures(*, dpi: int = 180) -> list[Path]:
    return [render_pipeline_figure(dpi=dpi), render_chemistry_cards_figure(dpi=dpi)]
