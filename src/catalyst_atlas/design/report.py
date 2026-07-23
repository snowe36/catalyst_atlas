"""Case-study report + README figures for shell redesign."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

from catalyst_atlas.design.pocket import load_pocket
from catalyst_atlas.paths import FIGURES, PROCESSED, REPORTS, ensure_dirs

logger = logging.getLogger(__name__)

TEAL = "#0E7490"
INK = "#1B2A2F"
MUTED = "#6B7280"
RULE = "#D1D5DB"
ACCENT = "#B45309"


def render_pocket_map(enzyme_id: str, path: Path | None = None) -> Path:
    """2D projection of catalytic (fixed) vs redesignable shell residues."""
    ensure_dirs()
    pocket = load_pocket(enzyme_id)
    out = path or (FIGURES / "fig_design_pocket_map.png")

    cat = pocket["catalytic_residues"]
    red = pocket["redesignable"]
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    if cat:
        cxyz = np.array([r["xyz"] for r in cat], dtype=float)
        ax.scatter(
            cxyz[:, 0],
            cxyz[:, 1],
            s=120,
            c=TEAL,
            marker="o",
            label="catalytic (fixed)",
            zorder=3,
            edgecolors=INK,
        )
        for r, (x, y, _) in zip(cat, cxyz, strict=True):
            ax.annotate(f"{r['aa']}{r['resnum']}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    if red:
        rxyz = np.array([r["xyz"] for r in red], dtype=float)
        shells = [r.get("shell") for r in red]
        colors = [ACCENT if s == "first" else MUTED for s in shells]
        ax.scatter(
            rxyz[:, 0],
            rxyz[:, 1],
            s=55,
            c=colors,
            marker="s",
            label="redesignable shell",
            alpha=0.85,
            zorder=2,
        )
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_title(f"{enzyme_id}: fixed catalytic core vs redesignable shell")
    ax.legend(frameon=False, loc="best")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def render_geometry_vs_wt(scores: pd.DataFrame, path: Path | None = None) -> Path:
    """Designs above/below WT geometry baseline."""
    ensure_dirs()
    out = path or (FIGURES / "fig_design_geometry_vs_wt.png")
    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    enzymes = sorted(scores["enzyme_id"].unique())
    for i, eid in enumerate(enzymes):
        sub = scores[scores["enzyme_id"] == eid]
        wt = sub[sub["is_wt"]]
        des = sub[~sub["is_wt"]]
        if wt.empty:
            continue
        wt_g = float(wt.iloc[0]["geometry_preservation"])
        xs = np.full(len(des), i) + np.linspace(-0.2, 0.2, max(len(des), 1))[: len(des)]
        ax.scatter(
            xs,
            des["geometry_preservation"],
            s=18,
            c=TEAL,
            alpha=0.65,
            edgecolors="none",
        )
        ax.hlines(wt_g, i - 0.35, i + 0.35, colors=ACCENT, linewidths=2, label="WT" if i == 0 else None)

    ax.axhline(1.0, color=RULE, linestyle="--", linewidth=0.8)
    ax.set_xticks(range(len(enzymes)))
    ax.set_xticklabels(enzymes, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("geometry_preservation")
    ax.set_title("Shell designs vs WT geometry baseline")
    ax.set_ylim(0.0, 1.05)
    if enzymes:
        ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def render_score_scatter(scores: pd.DataFrame, path: Path | None = None) -> Path:
    """geometry vs ESM, colored by structure confidence."""
    ensure_dirs()
    out = path or (FIGURES / "fig_design_score_scatter.png")
    des = scores[~scores["is_wt"]].copy()
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    if des.empty:
        ax.text(0.5, 0.5, "no designs", ha="center")
    else:
        sc = ax.scatter(
            des["esm_plausibility"],
            des["geometry_preservation"],
            c=des["structure_confidence"],
            cmap="viridis",
            s=28,
            alpha=0.85,
            edgecolors="none",
        )
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("structure_confidence")
    ax.set_xlabel("esm_plausibility")
    ax.set_ylabel("geometry_preservation")
    ax.set_title("Design ranking axes (proxies)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def render_design_landscape(
    scores: pd.DataFrame,
    path: Path | None = None,
    *,
    top_k_per_enzyme: int = 3,
) -> Path:
    """Geometry vs ESM landscape: WT, top candidates, rejected shortlist."""
    ensure_dirs()
    out = path or (FIGURES / "fig_design_landscape.png")
    fig, ax = plt.subplots(figsize=(6.8, 5.4))

    score_col = (
        "chemistry_constraint_score"
        if "chemistry_constraint_score" in scores.columns
        else "chemistry_preservation_score"
    )
    wt = scores[scores["is_wt"]]
    des = scores[~scores["is_wt"]].copy()
    top_ids: set[str] = set()
    if not des.empty and score_col in des.columns:
        for _, grp in des.groupby("enzyme_id"):
            top_ids.update(
                grp.sort_values(score_col, ascending=False)
                .head(top_k_per_enzyme)["design_id"]
                .astype(str)
            )
    top = des[des["design_id"].astype(str).isin(top_ids)]
    rejected = des[~des["design_id"].astype(str).isin(top_ids)]

    if not rejected.empty:
        ax.scatter(
            rejected["geometry_preservation"],
            rejected["esm_plausibility"],
            s=22,
            c=MUTED,
            alpha=0.4,
            edgecolors="none",
            label="rejected candidates",
            zorder=1,
        )
    if not top.empty:
        ax.scatter(
            top["geometry_preservation"],
            top["esm_plausibility"],
            s=52,
            c=TEAL,
            alpha=0.95,
            edgecolors=INK,
            linewidths=0.4,
            label="top candidates",
            zorder=2,
        )
    if not wt.empty:
        ax.scatter(
            wt["geometry_preservation"],
            wt["esm_plausibility"],
            s=120,
            c=ACCENT,
            marker="*",
            edgecolors=INK,
            linewidths=0.5,
            label="WT",
            zorder=3,
        )
        ax.axvline(float(wt["geometry_preservation"].mean()), color=RULE, ls="--", lw=0.9, zorder=0)
        ax.axhline(float(wt["esm_plausibility"].mean()), color=RULE, ls="--", lw=0.9, zorder=0)

    ax.set_xlabel("geometry (proxy)")
    ax.set_ylabel("ESM plausibility")
    ax.set_title("Design landscape — search space vs WT")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def _top_designs_table(scores: pd.DataFrame, k: int = 3) -> list[dict[str, Any]]:
    score_col = (
        "chemistry_constraint_score"
        if "chemistry_constraint_score" in scores.columns
        else "chemistry_preservation_score"
    )
    rows = []
    for eid, grp in scores.groupby("enzyme_id"):
        wt = grp[grp["is_wt"]]
        des = grp[~grp["is_wt"]].sort_values(score_col, ascending=False)
        wt_score = float(wt.iloc[0][score_col]) if len(wt) else float("nan")
        for _, r in des.head(k).iterrows():
            rows.append(
                {
                    "enzyme_id": eid,
                    "design_id": r["design_id"],
                    "chemistry_constraint_score": round(float(r[score_col]), 3),
                    "delta_vs_wt": round(float(r["delta_score_vs_wt"]), 3),
                    "geometry": round(float(r["geometry_preservation"]), 3),
                    "structure": round(float(r["structure_confidence"]), 3),
                    "esm": round(float(r["esm_plausibility"]), 3),
                    "n_mutations": int(r["n_mutations"]),
                    "mutations": str(r.get("mutations") or "")[:80],
                    "wt_score": round(wt_score, 3),
                    "chemistry_family": r.get("chemistry_family", ""),
                }
            )
    return rows


def write_design_case_study(
    scores: pd.DataFrame | None = None,
    panel: list[dict[str, Any]] | None = None,
) -> Path:
    ensure_dirs()
    if scores is None:
        path = PROCESSED / "design" / "scores.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run cat-design-score first")
        scores = pd.read_parquet(path)

    first_eid = sorted(scores["enzyme_id"].unique())[0]
    fig_pocket = render_pocket_map(first_eid)
    fig_landscape = render_design_landscape(scores)
    fig_geom = render_geometry_vs_wt(scores)
    fig_scatter = render_score_scatter(scores)

    def _rel(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(REPORTS.parent.resolve()))
        except ValueError:
            return str(path)

    fig_pocket_rel = _rel(fig_pocket)
    fig_landscape_rel = _rel(fig_landscape)
    fig_geom_rel = _rel(fig_geom)
    fig_scatter_rel = _rel(fig_scatter)

    top = _top_designs_table(scores, k=3)
    n_enz = scores["enzyme_id"].nunique()
    n_des = int((~scores["is_wt"]).sum())

    from catalyst_atlas.design.findings import build_findings, findings_markdown

    findings = build_findings(scores)

    lines = [
        "# Design case study — shell redesign with fixed catalytic chemistry",
        "",
        "**Question:** Can generative models optimize the molecular environment "
        "surrounding known catalytic machinery?",
        "",
        "Catalytic residues are held fixed; first-/second-shell positions are redesigned. "
        "Designs are ranked by `chemistry_constraint_score` "
        "(0.4 geometry + 0.3 structure confidence + 0.3 ESM plausibility) — "
        "**proxies for constraint satisfaction, not measured catalysis**.",
        "",
        "ProteinMPNN (or any sequence generator) proposes shell variants; the chemistry-aware "
        "evaluator — hard constraints, cheap ranking, AF geometry / confidence / ESM — decides "
        "which candidates are mechanistically and structurally plausible.",
        "",
        f"- Enzymes: **{n_enz}**",
        f"- Designs scored: **{n_des}**",
        f"- Pocket example figure: `{fig_pocket_rel}`",
        f"- Design landscape: `{fig_landscape_rel}`",
        f"- Geometry vs WT: `{fig_geom_rel}`",
        f"- Score scatter: `{fig_scatter_rel}`",
        "",
    ]
    lines.extend(findings_markdown(findings))
    lines.extend(
        [
        "## Panel",
        "",
        ]
    )
    if panel:
        lines.append("| enzyme_id | role | chemistry | redesignable |")
        lines.append("|---|---|---|---:|")
        for p in panel:
            lines.append(
                f"| `{p['enzyme_id']}` | {p.get('role', '')} | "
                f"{p.get('chemistry_family', '')} / {p.get('mechanistic_pattern', '')} | "
                f"{p.get('n_redesignable', '')} |"
            )
        lines.append("")
    else:
        for eid in sorted(scores["enzyme_id"].unique()):
            fam = scores.loc[scores["enzyme_id"] == eid, "chemistry_family"].iloc[0]
            lines.append(f"- `{eid}` — {fam}")
        lines.append("")

    lines.extend(
        [
            "## Top designs vs WT",
            "",
            "| enzyme | design | score | Δ vs WT | geometry | structure | ESM | mutations |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for r in top:
        lines.append(
            f"| `{r['enzyme_id']}` | `{r['design_id']}` | {r['chemistry_constraint_score']} | "
            f"{r['delta_vs_wt']:+.3f} | {r['geometry']} | {r['structure']} | {r['esm']} | "
            f"{r['n_mutations']} |"
        )

    funnel_meta_path = PROCESSED / "design" / "funnel_meta.json"
    if funnel_meta_path.exists():
        fm = json.loads(funnel_meta_path.read_text())
        lines.extend(
            [
                "",
                "## Design funnel",
                "",
                f"- Input designs: **{fm.get('n_input_designs')}**",
                f"- Passed hard filters: **{fm.get('n_passed_hard_filter')}**",
                f"- AF shortlist: **{fm.get('n_af_designs')}** designs + **{fm.get('n_af_wt')}** WT",
                "",
                "Story: chemistry-constrained funnel reduced generative candidates "
                "to an experimentally sized AF set — not 1000 structure jobs.",
                "",
            ]
        )

    lines.extend(
        [
            "",
            "## Method notes",
            "",
            "- Funnel: generate → hard filters → ESM + fixed-backbone chemistry → AF shortlist → mechanistic rank.",
            "- Generator and evaluation are separated (`generate` / `mpnn` vs `predict` / `score`).",
            "- The generator is a proposal mechanism, not an oracle.",
            "- Hard invariants: catalytic sequence identity; mutations ⊆ redesignable shell.",
            "- WT is scored with the same axes before any design comparison.",
            "- ProteinMPNN + ColabFold (reuse_wt MSA: WT MSA → design A3Ms); "
            "geometry = catalytic CA pairwise distances of design AF vs WT AF.",
            "- Scope ends at mechanistic ranking after AF — no MD in this repo.",
            "",
        ]
    )

    out = REPORTS / "design_case_study.md"
    out.write_text("\n".join(lines))
    summary = {
        "n_enzymes": n_enz,
        "n_designs": n_des,
        "figures": [fig_pocket_rel, fig_landscape_rel, fig_geom_rel, fig_scatter_rel],
        "top_designs": top,
        "findings": findings,
    }
    (REPORTS / "design_case_study_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Wrote design case study → %s", out)
    return out


def run_design_pipeline(
    *,
    target_size: int = 10,
    n_sequences: int = 100,
    mock: bool = True,
    seed: int = 7,
    top_k: int = 10,
    max_mutations: int = 12,
) -> dict[str, Any]:
    """End-to-end case study with chemistry-constrained AF funnel."""
    from catalyst_atlas.design.funnel import run_funnel
    from catalyst_atlas.design.generate import run_generate
    from catalyst_atlas.design.panel import resolve_panel
    from catalyst_atlas.design.pocket import run_pockets
    from catalyst_atlas.design.score import run_score

    panel = resolve_panel(target_size=target_size)
    eids = [p["enzyme_id"] for p in panel]
    run_pockets(enzyme_ids=eids)
    run_generate(eids, n_sequences=n_sequences, use_mock=mock, seed=seed)
    funnel_meta = run_funnel(top_k=top_k, max_mutations=max_mutations, enzyme_ids=eids)
    # Mechanistic rank only on AF shortlist (+ WT), not the full generative pool.
    scores = run_score(
        eids, mock_predictions=mock, seed=seed, af_queue_only=True
    )

    report = write_design_case_study(scores, panel=panel)
    return {
        "panel": panel,
        "n_designs": int((~scores["is_wt"]).sum()),
        "funnel": funnel_meta,
        "report": str(report),
    }
