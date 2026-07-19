"""Offline 3D figures of catalytic microenvironments (no PyMOL required).

Renders chemistry residues, first-shell neighbors, and cofactors from the
demo atlas using matplotlib — suitable for CI and README hero visuals.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from catalyst_atlas.paths import FIGURES, PROCESSED, REPORTS, ensure_dirs

logger = logging.getLogger(__name__)

PALETTE = {
    "bg": "#F4F7F6",
    "panel": "#E8EEEC",
    "ink": "#1B2A2F",
    "muted": "#5C6B73",
    "catalytic": "#C47A2C",  # amber catalytic core
    "catalytic_edge": "#0F6E6A",  # deep teal geometry
    "first_shell": "#7A8B94",  # muted slate
    "cofactor": "#0E7490",  # teal organic cofactors
    "metal": "#B45309",  # warm metal accent
    "grid": "#C5D0D4",
}


def _parse_micro(row: pd.Series) -> dict[str, list[dict[str, Any]]]:
    raw = row.get("microenvironment_json")
    if isinstance(raw, str):
        return json.loads(raw)
    if isinstance(raw, dict):
        return raw
    raise ValueError("Row is missing microenvironment_json")


def _xyz(points: list[dict[str, Any]]) -> np.ndarray:
    if not points:
        return np.zeros((0, 3), dtype=float)
    return np.array([p["xyz"] for p in points], dtype=float)


def _set_equal_aspect(ax, pts: np.ndarray, pad: float = 1.8) -> None:
    if pts.size == 0:
        return
    center = pts.mean(axis=0)
    radius = max(float(np.linalg.norm(pts - center, axis=1).max()), 3.0) + pad
    for setter, c in zip(
        (ax.set_xlim, ax.set_ylim, ax.set_zlim),
        center,
        strict=True,
    ):
        setter(c - radius, c + radius)


def _style_axes(ax, title: str, subtitle: str) -> None:
    ax.set_facecolor(PALETTE["panel"])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_edgecolor(PALETTE["grid"])
        axis.line.set_color(PALETTE["grid"])
    ax.tick_params(colors=PALETTE["muted"], labelsize=7, pad=1)
    ax.set_xlabel("x (Å)", color=PALETTE["muted"], fontsize=8, labelpad=2)
    ax.set_ylabel("y (Å)", color=PALETTE["muted"], fontsize=8, labelpad=2)
    ax.set_zlabel("z (Å)", color=PALETTE["muted"], fontsize=8, labelpad=2)
    ax.set_title(title, color=PALETTE["ink"], fontsize=12, fontweight="bold", pad=10)
    ax.text2D(
        0.02,
        0.02,
        subtitle,
        transform=ax.transAxes,
        color=PALETTE["muted"],
        fontsize=8,
        va="bottom",
    )
    ax.view_init(elev=18, azim=-58)


def _draw_catalytic_geometry(ax, catalytic: list[dict[str, Any]]) -> None:
    coords = _xyz(catalytic)
    if len(coords) < 2:
        return
    segments = []
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            segments.append([coords[i], coords[j]])
    coll = Line3DCollection(
        segments,
        colors=PALETTE["catalytic_edge"],
        linewidths=2.2,
        alpha=0.85,
        zorder=2,
    )
    ax.add_collection3d(coll)


def _draw_site(
    ax,
    catalytic: list[dict[str, Any]],
    first_shell: list[dict[str, Any]],
    ligands: list[dict[str, Any]],
    *,
    label_residues: bool = True,
) -> None:
    shell_xyz = _xyz(first_shell)
    cat_xyz = _xyz(catalytic)
    lig_xyz = _xyz(ligands)

    if len(shell_xyz):
        ax.scatter(
            shell_xyz[:, 0],
            shell_xyz[:, 1],
            shell_xyz[:, 2],
            s=70,
            c=PALETTE["first_shell"],
            alpha=0.45,
            depthshade=True,
            edgecolors="none",
            zorder=1,
        )

    _draw_catalytic_geometry(ax, catalytic)

    if len(cat_xyz):
        ax.scatter(
            cat_xyz[:, 0],
            cat_xyz[:, 1],
            cat_xyz[:, 2],
            s=220,
            c=PALETTE["catalytic"],
            alpha=0.95,
            depthshade=True,
            edgecolors=PALETTE["catalytic_edge"],
            linewidths=1.4,
            zorder=5,
        )
        if label_residues:
            for res, xyz in zip(catalytic, cat_xyz, strict=True):
                ax.text(
                    xyz[0],
                    xyz[1],
                    xyz[2] + 0.55,
                    f"{res['aa']}{res['resnum']}",
                    color=PALETTE["ink"],
                    fontsize=8,
                    fontweight="bold",
                    ha="center",
                    va="bottom",
                )

    for lig, xyz in zip(ligands, lig_xyz, strict=True):
        is_metal = lig.get("kind") == "metal" or lig.get("name") in {
            "Zn",
            "Fe",
            "Mg",
            "Mn",
        }
        color = PALETTE["metal"] if is_metal else PALETTE["cofactor"]
        marker = "D" if is_metal else "P"
        ax.scatter(
            [xyz[0]],
            [xyz[1]],
            [xyz[2]],
            s=260 if is_metal else 300,
            c=color,
            marker=marker,
            alpha=0.95,
            edgecolors=PALETTE["ink"],
            linewidths=0.8,
            zorder=6,
        )
        ax.text(
            xyz[0],
            xyz[1],
            xyz[2] + 0.7,
            str(lig["name"]),
            color=color,
            fontsize=8,
            fontweight="bold",
            ha="center",
        )
        # Soft contacts from cofactor to catalytic residues.
        for cxyz in cat_xyz:
            ax.plot(
                [xyz[0], cxyz[0]],
                [xyz[1], cxyz[1]],
                [xyz[2], cxyz[2]],
                color=color,
                alpha=0.25,
                linewidth=1.0,
                linestyle="--",
            )

    all_pts = (
        np.vstack([p for p in (cat_xyz, shell_xyz, lig_xyz) if len(p)])
        if any(len(p) for p in (cat_xyz, shell_xyz, lig_xyz))
        else np.zeros((1, 3))
    )
    _set_equal_aspect(ax, all_pts)


def _legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=PALETTE["catalytic"],
            markeredgecolor=PALETTE["catalytic_edge"],
            markersize=9,
            label="Catalytic residues",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=PALETTE["first_shell"],
            markersize=7,
            alpha=0.7,
            label="First shell",
        ),
        Line2D(
            [0],
            [0],
            marker="P",
            color="w",
            markerfacecolor=PALETTE["cofactor"],
            markeredgecolor=PALETTE["ink"],
            markersize=9,
            label="Cofactor",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor=PALETTE["metal"],
            markeredgecolor=PALETTE["ink"],
            markersize=8,
            label="Metal",
        ),
        Line2D(
            [0],
            [0],
            color=PALETTE["catalytic_edge"],
            linewidth=2.0,
            label="Catalytic geometry",
        ),
    ]


def render_microenvironment(
    row: pd.Series,
    out_path: Path,
    *,
    title: str | None = None,
    dpi: int = 180,
) -> Path:
    """Render one catalytic microenvironment to PNG."""
    ensure_dirs()
    micro = _parse_micro(row)
    catalytic = micro.get("catalytic") or []
    first_shell = micro.get("first_shell") or []
    ligands = micro.get("ligands") or []

    eid = str(row.get("enzyme_id", "enzyme"))
    chem = str(row.get("chemistry_class", ""))
    pattern = str(row.get("catalytic_pattern", ""))
    fig = plt.figure(figsize=(7.2, 5.6), facecolor=PALETTE["bg"])
    ax = fig.add_subplot(111, projection="3d")
    _draw_site(ax, catalytic, first_shell, ligands)
    _style_axes(
        ax,
        title or f"{eid} — catalytic microenvironment",
        f"{chem} · {pattern} · chemistry residues + first shell + cofactors",
    )
    fig.legend(
        handles=_legend_handles(),
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_gallery(
    rows: list[pd.Series],
    out_path: Path,
    *,
    dpi: int = 180,
) -> Path:
    """Side-by-side microenvironment panels for README support visuals."""
    ensure_dirs()
    n = len(rows)
    if n == 0:
        raise ValueError("No rows to render")
    fig = plt.figure(figsize=(4.2 * n, 4.8), facecolor=PALETTE["bg"])
    for i, row in enumerate(rows):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        micro = _parse_micro(row)
        _draw_site(
            ax,
            micro.get("catalytic") or [],
            micro.get("first_shell") or [],
            micro.get("ligands") or [],
            label_residues=True,
        )
        eid = str(row.get("enzyme_id", f"site-{i}"))
        chem = str(row.get("chemistry_class", ""))
        pattern = str(row.get("catalytic_pattern", ""))
        _style_axes(ax, eid, f"{chem} · {pattern}")
    fig.suptitle(
        "Catalytic microenvironments",
        color=PALETTE["ink"],
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.legend(
        handles=_legend_handles(),
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return out_path


def _load_micro_table(path: Path | None = None) -> pd.DataFrame:
    micro_path = path or (PROCESSED / "microenvironments.parquet")
    if not micro_path.exists():
        raise FileNotFoundError(
            f"Missing {micro_path}; run cat-download && cat-sites first"
        )
    return pd.read_parquet(micro_path)


def _hero_enzyme_id(df: pd.DataFrame) -> str | None:
    hero_json = PROCESSED / "hero_case.json"
    if hero_json.exists():
        payload = json.loads(hero_json.read_text())
        eid = payload.get("enzyme_id")
        if eid and eid in set(df["enzyme_id"]):
            return str(eid)
    # Fall back to the markdown report if present.
    md_path = REPORTS / "hero_cryptic_chemistry.md"
    if md_path.exists():
        for line in md_path.read_text().splitlines():
            if "Query enzyme:" in line and "`" in line:
                return line.split("`")[1]
    # Prefer a cryptic-seed lyase / oxidoreductase with a clear core.
    if "is_cryptic_seed" in df.columns:
        cryptic = df[df["is_cryptic_seed"].fillna(False).astype(bool)]
        pool = cryptic if len(cryptic) else df
    else:
        pool = df
    for pattern in ("Thr-Asp-His", "heme-redox", "Ser-His-Asp"):
        hit = pool[pool["catalytic_pattern"] == pattern]
        if len(hit):
            return str(hit.iloc[0]["enzyme_id"])
    return str(df.iloc[0]["enzyme_id"]) if len(df) else None


def _pick_by_pattern(df: pd.DataFrame, pattern: str) -> pd.Series | None:
    hits = df[df["catalytic_pattern"] == pattern]
    if hits.empty:
        return None
    # Prefer cofactor-bearing sites when available.
    with_cof = hits[hits["n_cofactors"] > 0] if "n_cofactors" in hits.columns else hits
    return (with_cof if len(with_cof) else hits).iloc[0]


def generate_structure_figures(
    micro_path: Path | None = None,
    out_dir: Path | None = None,
    *,
    dpi: int = 180,
) -> list[Path]:
    """Write README-ready catalytic microenvironment PNGs under reports/figures/."""
    ensure_dirs()
    df = _load_micro_table(micro_path)
    dest = Path(out_dir) if out_dir else FIGURES
    dest.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    hero_id = _hero_enzyme_id(df)
    if hero_id is not None:
        hero_row = df[df["enzyme_id"] == hero_id].iloc[0]
        path = render_microenvironment(
            hero_row,
            dest / "fig_microenv_hero_cryptic.png",
            title=f"{hero_id} — cryptic chemistry microenvironment",
            dpi=dpi,
        )
        written.append(path)
        logger.info("Wrote hero microenvironment figure → %s", path)

    showcase: list[tuple[str, str, str]] = [
        (
            "heme-redox",
            "fig_microenv_heme_redox.png",
            "Heme-redox catalytic core + Fe/heme contacts",
        ),
        (
            "Ser-His-Asp",
            "fig_microenv_catalytic_triad.png",
            "Ser-His-Asp catalytic triad microenvironment",
        ),
        (
            "Zn-activation",
            "fig_microenv_zn_activation.png",
            "Zn-activation metal–residue geometry",
        ),
    ]
    gallery_rows: list[pd.Series] = []
    for pattern, filename, title in showcase:
        row = _pick_by_pattern(df, pattern)
        if row is None:
            continue
        # Avoid duplicating the hero panel as a standalone showcase if identical.
        if hero_id and row["enzyme_id"] == hero_id and "hero" in filename:
            continue
        path = render_microenvironment(row, dest / filename, title=title, dpi=dpi)
        written.append(path)
        gallery_rows.append(row)
        logger.info("Wrote %s → %s", pattern, path)

    if len(gallery_rows) >= 2:
        # Prefer hero + up to two distinct chemistry showcases in the gallery.
        gallery: list[pd.Series] = []
        if hero_id is not None:
            gallery.append(df[df["enzyme_id"] == hero_id].iloc[0])
        for row in gallery_rows:
            if all(row["enzyme_id"] != g["enzyme_id"] for g in gallery):
                gallery.append(row)
            if len(gallery) >= 3:
                break
        if len(gallery) >= 2:
            gpath = render_gallery(gallery[:3], dest / "fig_microenv_gallery.png", dpi=dpi)
            written.append(gpath)
            logger.info("Wrote gallery → %s", gpath)

    if not written:
        raise RuntimeError("No structure figures generated; check microenvironments table")
    return written
