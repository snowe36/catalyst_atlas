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
        color = "#0E7490" if i == n - 1 else "#1B2A2F"
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
        color="#1B2A2F",
        pad=8,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Wrote pipeline figure → %s", out)
    return out


def render_chemistry_cards_figure(path: Path | None = None, *, dpi: int = 180) -> Path:
    """Figure 4 — example chemistry cards (convergent + same-fold contrast)."""
    ensure_dirs()
    out = Path(path) if path else FIGURES / "fig4_chemistry_cards.png"

    cards = [
        {
            "title": "Convergent chemistry",
            "query": "MCSA00176 · thermolysin",
            "context": "Remote sequence · distinct fold · shared Zn hydrolysis",
            "prediction": "hydrolysis / metal activation",
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
            "analog": "Fe-redox catalytic neighbors",
            "why": [
                "Fe cofactor at reaction center",
                "His-Asp catalytic pattern",
                "neighbors disagree with fold prior",
            ],
        },
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for ax, card in zip(axes, cards, strict=True):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        frame = FancyBboxPatch(
            (0.03, 0.04),
            0.94,
            0.92,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.5,
            edgecolor="#0E7490",
            facecolor="#F8FAFB",
        )
        ax.add_patch(frame)
        ax.text(0.08, 0.88, card["title"], fontsize=12, fontweight="bold", color="#0E7490")
        ax.text(0.08, 0.78, card["query"], fontsize=10, fontweight="bold", color="#1B2A2F")
        ax.text(0.08, 0.70, card["context"], fontsize=8.5, color="#4B5563")
        ax.text(0.08, 0.58, "Catalyst Atlas", fontsize=9, color="#6B7280")
        ax.text(0.08, 0.50, card["prediction"], fontsize=11, fontweight="bold", color="#0E7490")
        ax.text(0.08, 0.40, f"Top analog: {card['analog']}", fontsize=8.5, color="#374151")
        ax.text(0.08, 0.30, "Evidence:", fontsize=9, fontweight="bold", color="#1B2A2F")
        for i, line in enumerate(card["why"]):
            ax.text(0.10, 0.23 - i * 0.07, f"-  {line}", fontsize=8.5, color="#1B2A2F")

    fig.suptitle("Example chemistry cards", fontsize=12, color="#1B2A2F", y=0.98)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Wrote chemistry-card figure → %s", out)
    return out


def generate_readme_figures(*, dpi: int = 180) -> list[Path]:
    return [render_pipeline_figure(dpi=dpi), render_chemistry_cards_figure(dpi=dpi)]
