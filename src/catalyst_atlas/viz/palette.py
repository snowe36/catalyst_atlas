"""Atlas figure color palette (see atlas-figure-palette skill)."""

from __future__ import annotations

PALETTE = {
    "sage": "#8FB996",
    "teal": "#5FA8A8",
    "peach": "#F4A261",
    "coral": "#E76F51",
    "lavender": "#B8A9E8",
    "mustard": "#E9C46A",
    "navy": "#264653",
}

TEXT = PALETTE["navy"]
MUTED = "#5A6F74"
GRID = "#D9E2E0"
FACE = "#FFFFFF"
SPINE = "#A8B8B5"

# Semantic roles for catalyst figures
TEAL = PALETTE["teal"]
INK = TEXT
RULE = GRID
ACCENT = PALETTE["peach"]
CARD_BG = "#F7FAF9"
SERIES = [
    PALETTE["teal"],
    PALETTE["sage"],
    PALETTE["mustard"],
    PALETTE["peach"],
    PALETTE["coral"],
    PALETTE["lavender"],
]
