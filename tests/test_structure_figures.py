"""Offline catalytic microenvironment figure generation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from catalyst_atlas.data.download import download_atlas
from catalyst_atlas.site.extract import run_site_extraction
from catalyst_atlas.viz.structure_figures import (
    generate_structure_figures,
    render_microenvironment,
)


def _toy_micro_row() -> pd.Series:
    catalytic = [
        {"chain": "A", "resnum": 10, "aa": "S", "role": "catalytic", "xyz": [0.0, 0.0, 0.0]},
        {"chain": "A", "resnum": 35, "aa": "H", "role": "catalytic", "xyz": [3.5, 0.2, 0.1]},
        {"chain": "A", "resnum": 60, "aa": "D", "role": "catalytic", "xyz": [0.3, 3.8, -0.2]},
    ]
    shell = [
        {
            "chain": "A",
            "resnum": 200,
            "aa": "N",
            "role": "first_shell",
            "xyz": [2.0, 2.0, 2.0],
        }
    ]
    ligands = [{"name": "Zn", "kind": "metal", "xyz": [1.2, 1.0, 0.4]}]
    return pd.Series(
        {
            "enzyme_id": "TOYFIG",
            "chemistry_class": "hydrolase",
            "catalytic_pattern": "Zn-activation",
            "n_cofactors": 1,
            "microenvironment_json": json.dumps(
                {"catalytic": catalytic, "first_shell": shell, "ligands": ligands}
            ),
        }
    )


def test_render_microenvironment_writes_png(tmp_path: Path):
    out = tmp_path / "toy_microenv.png"
    path = render_microenvironment(_toy_micro_row(), out, dpi=72)
    assert path.exists()
    assert path.stat().st_size > 1000


def test_render_metal_focus_writes_png(tmp_path: Path):
    out = tmp_path / "toy_zn.png"
    path = render_microenvironment(_toy_micro_row(), out, dpi=72, metal_focus=True)
    assert path.exists()
    assert path.stat().st_size > 1000


def test_generate_structure_figures_pipeline(isolated_data_dirs: Path):
    download_atlas(demo=True, n_enzymes=120, seed=7)
    run_site_extraction()

    import catalyst_atlas.paths as paths

    micro = pd.read_parquet(paths.PROCESSED / "microenvironments.parquet")
    hero_id = str(micro.iloc[0]["enzyme_id"])
    (paths.PROCESSED / "hero_case.json").write_text(
        json.dumps(
            {
                "enzyme_id": hero_id,
                "predicted": "hydrolase",
                "true": "hydrolase",
                "seq_baseline": "lyase",
                "fold_baseline": "hydrolase",
            }
        )
    )
    out_dir = isolated_data_dirs / "out" / "figures"
    paths_out = generate_structure_figures(out_dir=out_dir, dpi=72)
    assert paths_out
    names = {p.name for p in paths_out}
    assert "fig_microenv_hero_cryptic.png" in names
    assert any(n.startswith("fig_microenv_") and n.endswith(".png") for n in names)
    for p in paths_out:
        assert p.exists() and p.stat().st_size > 1000
