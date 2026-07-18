"""Offline catalytic microenvironment figure generation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from catalyst_atlas.data.download import download_atlas
from catalyst_atlas.paths import FIGURES, PROCESSED
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


def test_generate_structure_figures_pipeline(tmp_path: Path, monkeypatch):
    download_atlas(demo=True, n_enzymes=120, seed=7)
    run_site_extraction()
    # Point figure output at an isolated directory; inputs still use package paths.
    out_dir = tmp_path / "figures"
    # Seed a hero case so the cryptic figure targets a known enzyme.
    micro = pd.read_parquet(PROCESSED / "microenvironments.parquet")
    hero_id = str(micro.iloc[0]["enzyme_id"])
    (PROCESSED / "hero_case.json").write_text(
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
    paths = generate_structure_figures(out_dir=out_dir, dpi=72)
    assert paths
    names = {p.name for p in paths}
    assert "fig_microenv_hero_cryptic.png" in names
    assert any(n.startswith("fig_microenv_") and n.endswith(".png") for n in names)
    for p in paths:
        assert p.exists() and p.stat().st_size > 1000
    # Package FIGURES dir should remain usable for regenerate docs.
    assert FIGURES.exists()
