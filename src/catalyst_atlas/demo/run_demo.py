"""Self-contained demo: synthetic atlas + mock design (no M-CSA / ColabFold)."""

from __future__ import annotations

import shutil
from pathlib import Path

from catalyst_atlas.paths import FIGURES, ROOT, ensure_dirs

DEMO = ROOT / "demo"
DEMO_OUT = DEMO / "outputs"

_HERO_FIGS = [
    "fig1_pipeline.png",
    "fig_design_pocket_map.png",
    "fig_design_landscape.png",
    "fig_design_score_scatter.png",
    "fig_design_geometry_vs_wt.png",
    "fig4_chemistry_cards.png",
    "fig_microenv_gallery.png",
    "fig_fold_disconnected_chemistry.png",
    "fig_retrieval_neighbors.png",
]


def run_demo() -> dict:
    """Build a small demo atlas, run mock redesign, stage demo/outputs/."""
    import os

    os.environ.setdefault("MPLBACKEND", "Agg")
    ensure_dirs()
    DEMO_OUT.mkdir(parents=True, exist_ok=True)

    from catalyst_atlas.cli import design_run_main, download_main, embed_main, sites_main
    from catalyst_atlas.viz.readme_figures import generate_readme_figures

    print("=== catalyst_atlas demo (synthetic atlas + mock design) ===")
    download_main(["--demo", "--n-enzymes", "120"])
    sites_main([])
    embed_main([])
    design_run_main(
        ["--mock", "--panel-size", "4", "--n-sequences", "40", "--top-k", "5"]
    )
    # README cards/pipeline always work offline; full structure gallery needs M-CSA heroes.
    try:
        generate_readme_figures()
    except Exception as exc:  # noqa: BLE001
        print(f"readme figures skipped: {exc}")
    try:
        from catalyst_atlas.cli import figures_main

        figures_main([])
    except Exception as exc:  # noqa: BLE001
        print(f"structure gallery skipped (demo atlas lacks M-CSA heroes): {exc}")

    copied = []
    for name in _HERO_FIGS:
        src = FIGURES / name
        if src.exists():
            shutil.copy2(src, DEMO_OUT / name)
            copied.append(name)

    report = ROOT / "out" / "design_case_study.md"
    if report.exists():
        shutil.copy2(report, DEMO_OUT / report.name)
        copied.append(report.name)

    summary = {"outputs": str(DEMO_OUT), "copied": copied}
    print("")
    print(f"Demo outputs → {DEMO_OUT}")
    for c in sorted(set(copied)):
        print(f"  - {c}")
    print("CPU mock design only — proxies ≠ measured catalysis.")
    return summary
