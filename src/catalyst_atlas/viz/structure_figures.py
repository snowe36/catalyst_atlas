"""Offline 3D figures of catalytic microenvironments (no PyMOL required).

Renders chemistry residues, first-shell neighbors, and cofactors from the
atlas using matplotlib — suitable for CI and README hero visuals.
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
    "bg": "#FFFFFF",
    "panel": "#FAFBFC",
    "ink": "#0F172A",
    "muted": "#64748B",
    "catalytic": "#0F766E",  # teal — catalytic residues
    "catalytic_edge": "#99F6E4",  # soft geometry between residues
    "first_shell": "#CBD5E1",  # quiet slate
    "cofactor": "#0369A1",  # blue organic cofactors
    "metal": "#EA580C",  # vivid orange metal (high contrast)
    "coord": "#F59E0B",  # metal–residue coordination
    "grid": "#E2E8F0",
}

METALS = {"Zn", "Fe", "Mg", "Mn", "Cu", "Ni", "Co", "Ca"}


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


def _set_equal_aspect(ax, pts: np.ndarray, pad: float = 3.0) -> None:
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


def _best_view(points: np.ndarray) -> tuple[float, float]:
    """Pick elev/azim that spreads labeled points and avoids depth stacking."""
    if len(points) < 2:
        return 18.0, 35.0
    centered = points - points.mean(axis=0)
    best = (18.0, 35.0)
    best_score = -1.0
    for elev in (8, 14, 20, 28, 36, -10, -18):
        for azim in range(0, 360, 10):
            er = np.radians(elev)
            ar = np.radians(azim)
            # Matplotlib-style projection onto viewing plane + depth.
            x = centered[:, 0] * np.cos(ar) + centered[:, 1] * np.sin(ar)
            y = (
                -centered[:, 0] * np.sin(ar) * np.sin(er)
                + centered[:, 1] * np.cos(ar) * np.sin(er)
                + centered[:, 2] * np.cos(er)
            )
            depth = (
                centered[:, 0] * np.sin(ar) * np.cos(er)
                - centered[:, 1] * np.cos(ar) * np.cos(er)
                + centered[:, 2] * np.sin(er)
            )
            proj = np.column_stack([x, y])
            pair_scores = []
            for i in range(len(proj)):
                for j in range(i + 1, len(proj)):
                    d2 = float(np.linalg.norm(proj[i] - proj[j]))
                    # Penalize near-overlaps that are separated only in depth.
                    if d2 < 1.2 and abs(float(depth[i] - depth[j])) > 0.8:
                        d2 *= 0.25
                    pair_scores.append(d2)
            score = min(pair_scores) if pair_scores else 0.0
            # Prefer a modest elevation so coordination geometry reads in depth.
            score *= 1.0 + 0.05 * (1.0 - abs(abs(elev) - 18) / 18.0)
            if score > best_score:
                best_score = score
                best = (float(elev), float(azim))
    return best


def _style_axes(
    ax,
    title: str,
    subtitle: str,
    *,
    view_points: np.ndarray | None = None,
) -> None:
    ax.set_facecolor(PALETTE["panel"])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_edgecolor("#F8FAFC")
        axis.pane.set_alpha(0.15)
        axis.line.set_color("#CBD5E1")
        axis._axinfo["grid"]["color"] = "#EEF2F7"
        axis._axinfo["grid"]["linewidth"] = 0.25
        axis._axinfo["grid"]["linestyle"] = "-"
    ax.grid(True, alpha=0.25)
    ax.tick_params(colors=PALETTE["muted"], labelsize=7, pad=0)
    ax.set_xlabel("x (Å)", color=PALETTE["muted"], fontsize=8, labelpad=1)
    ax.set_ylabel("y (Å)", color=PALETTE["muted"], fontsize=8, labelpad=1)
    ax.set_zlabel("z (Å)", color=PALETTE["muted"], fontsize=8, labelpad=1)
    ax.set_title(title, color=PALETTE["ink"], fontsize=12, fontweight="bold", pad=8)
    ax.text2D(
        0.02,
        -0.02,
        subtitle,
        transform=ax.transAxes,
        color=PALETTE["muted"],
        fontsize=8,
        va="top",
    )
    elev, azim = _best_view(view_points) if view_points is not None else (18.0, 35.0)
    ax.view_init(elev=elev, azim=azim)


def _label_offsets(n: int) -> list[np.ndarray]:
    """Spread residue labels so they do not stack on the metal."""
    if n <= 0:
        return []
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + 0.35
    return [
        np.array([0.85 * np.cos(a), 0.85 * np.sin(a), 0.55 + 0.15 * (i % 2)], dtype=float)
        for i, a in enumerate(angles)
    ]


def _draw_catalytic_geometry(ax, catalytic: list[dict[str, Any]]) -> None:
    """Light edges between catalytic residues (secondary to metal bonds)."""
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
        linewidths=1.4,
        alpha=0.55,
        zorder=2,
    )
    ax.add_collection3d(coll)


def _nearest_shell(first_shell: list[dict[str, Any]], core: np.ndarray, k: int = 6) -> list[dict[str, Any]]:
    if not first_shell or core.size == 0:
        return first_shell
    center = core.mean(axis=0)
    ranked = sorted(
        first_shell,
        key=lambda r: float(np.linalg.norm(np.asarray(r["xyz"], dtype=float) - center)),
    )
    return ranked[:k]


def _primary_metal(ligands: list[dict[str, Any]]) -> dict[str, Any] | None:
    metals = [
        lig
        for lig in ligands
        if lig.get("kind") == "metal" or str(lig.get("name") or "") in METALS
    ]
    if not metals:
        return None
    # Prefer Zn for activation showcase; otherwise nearest-to-origin is fine.
    zn = [m for m in metals if str(m.get("name") or "") == "Zn" or str(m.get("het") or "") == "ZN"]
    return zn[0] if zn else metals[0]


def _focus_catalytic_for_metal(
    catalytic: list[dict[str, Any]],
    metal: dict[str, Any] | None,
    *,
    k: int = 4,
    max_dist: float = 7.5,
) -> list[dict[str, Any]]:
    """Keep the coordinating shell readable — drop distant annotated residues."""
    if not catalytic:
        return []
    if metal is None:
        return catalytic[: min(k, len(catalytic))]
    mxyz = np.asarray(metal["xyz"], dtype=float)
    ranked = sorted(
        catalytic,
        key=lambda r: float(np.linalg.norm(np.asarray(r["xyz"], dtype=float) - mxyz)),
    )
    near = [
        r
        for r in ranked
        if float(np.linalg.norm(np.asarray(r["xyz"], dtype=float) - mxyz)) <= max_dist
    ]
    chosen = (near or ranked)[:k]
    return chosen


def _label_pos_beyond_residue(
    res_xyz: np.ndarray,
    metal_xyz: np.ndarray | None,
    *,
    pad: float = 2.0,
) -> np.ndarray:
    """Place a residue label past the residue along the metal→residue ray."""
    if metal_xyz is None:
        return res_xyz + np.array([0.0, 0.0, pad], dtype=float)
    radial = res_xyz - metal_xyz
    norm = float(np.linalg.norm(radial))
    if norm < 1e-3:
        return res_xyz + np.array([0.0, 0.0, pad], dtype=float)
    return metal_xyz + radial * ((norm + pad) / norm)


def _draw_site(
    ax,
    catalytic: list[dict[str, Any]],
    first_shell: list[dict[str, Any]],
    ligands: list[dict[str, Any]],
    *,
    label_residues: bool = True,
    metal_focus: bool = False,
) -> None:
    metal = _primary_metal(ligands) if metal_focus else None
    display_cat = (
        _focus_catalytic_for_metal(catalytic, metal, k=3, max_dist=7.5)
        if metal_focus
        else catalytic
    )
    # For metal-centric figures, only keep the primary metal (+ organic cofactors).
    if metal_focus and metal is not None:
        organics = [
            lig
            for lig in ligands
            if not (lig.get("kind") == "metal" or str(lig.get("name") or "") in METALS)
        ]
        ligands = [metal, *organics]

    cat_xyz = _xyz(display_cat)
    shell = (
        []
        if metal_focus
        else _nearest_shell(first_shell, cat_xyz if len(cat_xyz) else _xyz(catalytic), k=5)
    )
    shell_xyz = _xyz(shell)
    lig_xyz = _xyz(ligands)
    metal_xyz = np.asarray(metal["xyz"], dtype=float) if metal is not None else None
    label_pts: list[np.ndarray] = []

    if len(shell_xyz):
        ax.scatter(
            shell_xyz[:, 0],
            shell_xyz[:, 1],
            shell_xyz[:, 2],
            s=40,
            c=PALETTE["first_shell"],
            alpha=0.28,
            depthshade=False,
            edgecolors="none",
            zorder=1,
        )

    # Soft residue–residue edges for non-metal figures only — metal sites lead with coordination.
    if not metal_focus and len(display_cat) <= 5:
        _draw_catalytic_geometry(ax, display_cat)

    if len(cat_xyz):
        ax.scatter(
            cat_xyz[:, 0],
            cat_xyz[:, 1],
            cat_xyz[:, 2],
            s=220,
            c=PALETTE["catalytic"],
            alpha=1.0,
            depthshade=False,
            edgecolors="#042F2E",
            linewidths=1.2,
            zorder=5,
        )
        if label_residues:
            for res, xyz in zip(display_cat, cat_xyz, strict=True):
                lpos = _label_pos_beyond_residue(xyz, metal_xyz, pad=4.0)
                label_pts.append(lpos)
                ax.text(
                    lpos[0],
                    lpos[1],
                    lpos[2],
                    f"{res['aa']}{res['resnum']}",
                    color=PALETTE["ink"],
                    fontsize=10,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    zorder=8,
                    bbox={
                        "boxstyle": "round,pad=0.15",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.75,
                    },
                )

    for lig, xyz in zip(ligands, lig_xyz, strict=True):
        is_metal = lig.get("kind") == "metal" or str(lig.get("name") or "") in METALS
        color = PALETTE["metal"] if is_metal else PALETTE["cofactor"]
        marker = "D" if is_metal else "P"
        ax.scatter(
            [xyz[0]],
            [xyz[1]],
            [xyz[2]],
            s=360 if is_metal else 220,
            c=color,
            marker=marker,
            alpha=1.0,
            edgecolors=PALETTE["ink"],
            linewidths=1.2,
            depthshade=False,
            zorder=7,
        )
        # Metal label: opposite the residue centroid so it stays clear.
        # Metal identity is clear from the orange diamond + legend; skip an
        # on-marker "Zn" label that crowds the coordination center.
        if not is_metal:
            label_xyz = xyz + np.array([0.0, 0.0, 0.9])
            label_pts.append(label_xyz)
            ax.text(
                label_xyz[0],
                label_xyz[1],
                label_xyz[2],
                str(lig["name"]),
                color=color,
                fontsize=8,
                fontweight="bold",
                ha="center",
                zorder=9,
            )
        if len(cat_xyz):
            dists = np.linalg.norm(cat_xyz - xyz, axis=1)
            order = np.argsort(dists)
            n_bonds = len(order) if is_metal else min(2, len(order))
            for idx in order[:n_bonds]:
                cxyz = cat_xyz[idx]
                ax.plot(
                    [xyz[0], cxyz[0]],
                    [xyz[1], cxyz[1]],
                    [xyz[2], cxyz[2]],
                    color=PALETTE["coord"] if is_metal else color,
                    alpha=0.98 if is_metal else 0.35,
                    linewidth=3.0 if is_metal else 1.0,
                    linestyle="-" if is_metal else "--",
                    solid_capstyle="round",
                    zorder=4,
                )

    all_pts = (
        np.vstack([p for p in (cat_xyz, shell_xyz, lig_xyz) if len(p)])
        if any(len(p) for p in (cat_xyz, shell_xyz, lig_xyz))
        else np.zeros((1, 3))
    )
    _set_equal_aspect(ax, all_pts)
    # Include label anchors so the camera separates text as well as atoms.
    focus = [p for p in (cat_xyz, lig_xyz) if len(p)]
    if label_pts:
        focus.append(np.asarray(label_pts, dtype=float))
    ax._atlas_view_points = np.vstack(focus) if focus else all_pts  # noqa: SLF001


def _legend_handles(*, metal_focus: bool = False) -> list[Line2D]:
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=PALETTE["catalytic"],
            markeredgecolor="#042F2E",
            markersize=9,
            label="Catalytic residues",
        ),
    ]
    if not metal_focus:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=PALETTE["first_shell"],
                markersize=7,
                alpha=0.7,
                label="First shell",
            )
        )
    handles.extend(
        [
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
                color=PALETTE["coord"],
                linewidth=2.2,
                label="Zn–residue distance" if metal_focus else "Metal coordination",
            ),
        ]
    )
    if not metal_focus:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="P",
                color="w",
                markerfacecolor=PALETTE["cofactor"],
                markeredgecolor=PALETTE["ink"],
                markersize=9,
                label="Organic cofactor",
            )
        )
    return handles


def _project_metal_shell(
    metal_xyz: np.ndarray,
    res_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Orthographic projection of metal + residues onto their best-fit plane."""
    pts = np.vstack([metal_xyz.reshape(1, 3), res_xyz])
    centered = pts - metal_xyz
    if len(pts) < 3:
        # Degenerate: drop the smallest-variance axis.
        var = centered.var(axis=0)
        keep = [i for i in np.argsort(var)[::-1][:2]]
        xy = centered[:, keep]
        return xy[0], xy[1:]
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2]
    xy = centered @ basis.T
    # Rotate so the residue centroid sits on +y (readable upright layout).
    if len(xy) > 1:
        centroid = xy[1:].mean(axis=0)
        angle = np.arctan2(centroid[0], centroid[1])
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, -s], [s, c]], dtype=float)
        xy = xy @ rot.T
    return xy[0], xy[1:]


def _render_metal_coordination_2d(
    row: pd.Series,
    out_path: Path,
    *,
    title: str | None = None,
    dpi: int = 180,
) -> Path:
    """Flat coordination schematic — clearer than matplotlib 3D for Zn sites."""
    ensure_dirs()
    micro = _parse_micro(row)
    catalytic = micro.get("catalytic") or []
    ligands = micro.get("ligands") or []
    metal = _primary_metal(ligands)
    if metal is None:
        raise ValueError("metal_focus figure requires a metal ligand")

    focused = _focus_catalytic_for_metal(catalytic, metal, k=3, max_dist=7.5)
    if not focused:
        raise ValueError("No catalytic residues near metal for coordination figure")

    metal_xyz = np.asarray(metal["xyz"], dtype=float)
    res_xyz = _xyz(focused)
    true_dists = np.linalg.norm(res_xyz - metal_xyz, axis=1)
    _, res_proj = _project_metal_shell(metal_xyz, res_xyz)
    # Keep projected angles, but place residues at true 3D distances so the
    # schematic scale matches the Å annotations.
    res_2d = np.zeros_like(res_proj)
    for i, (proj, dist) in enumerate(zip(res_proj, true_dists, strict=True)):
        nrm = float(np.linalg.norm(proj))
        if nrm < 1e-6:
            angle = 2.0 * np.pi * i / max(len(res_proj), 1)
            unit = np.array([np.sin(angle), np.cos(angle)], dtype=float)
        else:
            unit = proj / nrm
        res_2d[i] = unit * float(dist)

    eid = str(row.get("enzyme_id", "enzyme"))
    chem = str(row.get("chemistry_class", "") or row.get("chemistry_family", ""))
    shell_txt = "–".join(f"{r['aa']}{r['resnum']}" for r in focused)
    metal_name = str(metal.get("name") or "Zn")

    fig, ax = plt.subplots(figsize=(6.6, 6.2), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["panel"])

    # Soft coordination-sphere guide at mean ligand distance.
    mean_r = float(true_dists.mean())
    guide = plt.Circle(
        (0.0, 0.0),
        mean_r,
        fill=False,
        linestyle=(0, (2, 3)),
        linewidth=1.0,
        color="#CBD5E1",
        alpha=0.9,
        zorder=1,
    )
    ax.add_patch(guide)

    for (x, y), dist in zip(res_2d, true_dists, strict=True):
        ax.plot(
            [0.0, x],
            [0.0, y],
            color=PALETTE["coord"],
            linewidth=2.8,
            solid_capstyle="round",
            zorder=2,
        )
        bond = np.array([x, y], dtype=float)
        mid = 0.55 * bond
        bn = float(np.linalg.norm(bond))
        perp = (
            np.array([-bond[1], bond[0]], dtype=float) / bn
            if bn > 1e-6
            else np.array([0.0, 1.0])
        )
        tpos = mid + perp * 0.7
        ax.text(
            tpos[0],
            tpos[1],
            f"{dist:.1f} Å",
            color=PALETTE["muted"],
            fontsize=8,
            ha="center",
            va="center",
            zorder=5,
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": PALETTE["panel"],
                "edgecolor": "none",
                "alpha": 0.9,
            },
        )

    ax.scatter(
        res_2d[:, 0],
        res_2d[:, 1],
        s=520,
        c=PALETTE["catalytic"],
        edgecolors="#042F2E",
        linewidths=1.4,
        zorder=4,
    )
    ax.scatter(
        [0.0],
        [0.0],
        s=700,
        c=PALETTE["metal"],
        marker="D",
        edgecolors=PALETTE["ink"],
        linewidths=1.3,
        zorder=6,
    )
    ax.text(
        0.0,
        0.0,
        metal_name,
        color="white",
        fontsize=11,
        fontweight="bold",
        ha="center",
        va="center",
        zorder=7,
    )

    for (x, y), res in zip(res_2d, focused, strict=True):
        vec = np.array([x, y], dtype=float)
        nrm = float(np.linalg.norm(vec))
        unit = vec / nrm if nrm > 1e-6 else np.array([0.0, 1.0])
        lx, ly = unit * (nrm + 1.55)
        ax.text(
            lx,
            ly,
            f"{res['aa']}{res['resnum']}",
            color=PALETTE["ink"],
            fontsize=12,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=8,
        )

    # Compact scale bar.
    span = max(float(true_dists.max()) + 2.8, 6.0)
    bar_y = -span + 0.9
    bar_x0 = -1.0
    ax.plot([bar_x0, bar_x0 + 2.0], [bar_y, bar_y], color=PALETTE["ink"], linewidth=2.0)
    ax.text(
        bar_x0 + 1.0,
        bar_y - 0.45,
        "2 Å",
        color=PALETTE["muted"],
        fontsize=8,
        ha="center",
        va="top",
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(
        title or f"{eid} — metal coordination",
        color=PALETTE["ink"],
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax.text(
        0.5,
        -0.02,
        f"{chem} · {metal_name} · {shell_txt} · {eid}",
        transform=ax.transAxes,
        color=PALETTE["muted"],
        fontsize=8,
        ha="center",
        va="top",
    )
    fig.legend(
        handles=_legend_handles(metal_focus=True),
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_microenvironment(
    row: pd.Series,
    out_path: Path,
    *,
    title: str | None = None,
    dpi: int = 180,
    metal_focus: bool = False,
) -> Path:
    """Render one catalytic microenvironment to PNG."""
    if metal_focus:
        return _render_metal_coordination_2d(row, out_path, title=title, dpi=dpi)

    ensure_dirs()
    micro = _parse_micro(row)
    catalytic = micro.get("catalytic") or []
    first_shell = micro.get("first_shell") or []
    ligands = micro.get("ligands") or []

    eid = str(row.get("enzyme_id", "enzyme"))
    chem = str(row.get("chemistry_class", "") or row.get("chemistry_family", ""))
    pattern = str(row.get("catalytic_pattern", "") or row.get("mechanistic_pattern", ""))
    fig = plt.figure(figsize=(7.4, 5.8), facecolor=PALETTE["bg"])
    ax = fig.add_subplot(111, projection="3d")
    _draw_site(
        ax,
        catalytic,
        first_shell,
        ligands,
        metal_focus=False,
    )
    view_pts = getattr(ax, "_atlas_view_points", None)
    _style_axes(
        ax,
        title or f"{eid} — catalytic microenvironment",
        f"{chem} · {pattern}",
        view_points=view_pts,
    )
    fig.legend(
        handles=_legend_handles(metal_focus=False),
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
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
        chem = str(row.get("chemistry_class", "") or row.get("chemistry_family", ""))
        pattern = str(row.get("catalytic_pattern", "") or row.get("mechanistic_pattern", ""))
        _style_axes(
            ax,
            eid,
            f"{chem} · {pattern}",
            view_points=getattr(ax, "_atlas_view_points", None),
        )
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
    md_path = REPORTS / "hero_cryptic_chemistry.md"
    if md_path.exists():
        for line in md_path.read_text().splitlines():
            if "Query enzyme:" in line and "`" in line:
                return line.split("`")[1]
    if "is_cryptic_seed" in df.columns:
        cryptic = df[df["is_cryptic_seed"].fillna(False).astype(bool)]
        pool = cryptic if len(cryptic) else df
    else:
        pool = df
    for pattern in ("Thr-Asp-His", "heme-redox", "Ser-His-Asp", "Zn-activation"):
        hit = pool[pool["catalytic_pattern"] == pattern]
        if len(hit):
            return str(hit.iloc[0]["enzyme_id"])
    return str(df.iloc[0]["enzyme_id"]) if len(df) else None


def _score_zn_showcase(row: pd.Series) -> float | None:
    """Higher is better: single Zn, compact His/Glu/Asp shell, hydrolase preferred."""
    try:
        micro = _parse_micro(row)
    except ValueError:
        return None
    catalytic = micro.get("catalytic") or []
    ligands = micro.get("ligands") or []
    zn = [
        lig
        for lig in ligands
        if str(lig.get("name") or "") == "Zn" or str(lig.get("het") or "") == "ZN"
    ]
    if len(zn) != 1:
        return None
    if str(row.get("mechanistic_pattern") or "") != "metal activation":
        return None
    mxyz = np.asarray(zn[0]["xyz"], dtype=float)
    near = []
    for res in catalytic:
        dist = float(np.linalg.norm(np.asarray(res["xyz"], dtype=float) - mxyz))
        if res.get("aa") in {"H", "E", "D"} and dist <= 7.5:
            near.append(dist)
    if len(near) < 3:
        return None
    # Classic Zn figure reads best as a 3-residue coordination shell.
    triad_bonus = 1.35 if len(near) == 3 else (1.1 if len(near) == 4 else 1.0)
    near = sorted(near)[:4]
    compactness = 1.0 / (1.0 + (near[-1] - near[0]))
    closeness = 1.0 / (1.0 + near[0])
    # Prefer sites that are already small so we are not over-filtering.
    size_bonus = 1.0 / (1.0 + max(0, len(catalytic) - 4))
    hydrolase = 1.25 if str(row.get("chemistry_class") or "") == "hydrolase" else 1.0
    # Prefer His-rich Zn sites (classic activation geometry).
    his_n = sum(1 for res in catalytic if res.get("aa") == "H" and float(np.linalg.norm(np.asarray(res["xyz"], dtype=float) - mxyz)) <= 7.5)
    his_bonus = 1.0 + 0.08 * min(his_n, 3)
    return (compactness + closeness + size_bonus) * hydrolase * triad_bonus * his_bonus


def _pick_zn_showcase(df: pd.DataFrame) -> pd.Series | None:
    scored: list[tuple[float, int]] = []
    for idx, row in df.iterrows():
        score = _score_zn_showcase(row)
        if score is not None:
            scored.append((score, idx))
    if not scored:
        return None
    scored.sort(reverse=True)
    return df.loc[scored[0][1]]


def _pick_by_pattern(df: pd.DataFrame, pattern: str) -> pd.Series | None:
    if pattern == "Zn-activation":
        zn = _pick_zn_showcase(df)
        if zn is not None:
            return zn
    hits = df[df["catalytic_pattern"] == pattern]
    if hits.empty:
        # Fall back: mechanistic pattern / chemistry family heuristics.
        if pattern == "Zn-activation" and "mechanistic_pattern" in df.columns:
            hits = df[df["mechanistic_pattern"].astype(str).str.contains("metal", case=False)]
            if "cofactor_tags" in df.columns:
                zn_hits = hits[hits["cofactor_tags"].astype(str).str.contains("Zn", case=False)]
                hits = zn_hits if len(zn_hits) else hits
        if hits.empty:
            return None
    with_cof = hits[hits["n_cofactors"] > 0] if "n_cofactors" in hits.columns else hits
    return (with_cof if len(with_cof) else hits).iloc[0]


def generate_structure_figures(
    micro_path: Path | None = None,
    out_dir: Path | None = None,
    *,
    dpi: int = 180,
) -> list[Path]:
    """Write README-ready catalytic microenvironment PNGs under out/figures/."""
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
        if hero_id and row["enzyme_id"] == hero_id and "hero" in filename:
            continue
        path = render_microenvironment(
            row,
            dest / filename,
            title=title,
            dpi=dpi,
            metal_focus=(pattern == "Zn-activation"),
        )
        written.append(path)
        gallery_rows.append(row)
        logger.info("Wrote %s → %s", pattern, path)

    if len(gallery_rows) >= 2:
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
