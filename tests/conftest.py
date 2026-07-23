"""Shared fixtures — isolate pipeline I/O under tmp_path."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

# Modules that bind RAW / PROCESSED / FIGURES / REPORTS at import time.
_PATH_MODULES = (
    "catalyst_atlas.paths",
    "catalyst_atlas.data.generate_demo",
    "catalyst_atlas.site.extract",
    "catalyst_atlas.featurize.features",
    "catalyst_atlas.models.embed",
    "catalyst_atlas.eval.run",
    "catalyst_atlas.eval.external_baselines",
    "catalyst_atlas.search",
    "catalyst_atlas.viz.structure_figures",
)


@pytest.fixture
def isolated_data_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect package data/report paths into an isolated temp tree."""
    raw = tmp_path / "data" / "raw"
    processed = tmp_path / "data" / "processed"
    reports = tmp_path / "out"
    figures = reports / "figures"
    for path in (raw, processed, figures):
        path.mkdir(parents=True, exist_ok=True)

    for modname in _PATH_MODULES:
        mod = importlib.import_module(modname)
        if hasattr(mod, "ROOT"):
            monkeypatch.setattr(mod, "ROOT", tmp_path)
        if hasattr(mod, "DATA"):
            monkeypatch.setattr(mod, "DATA", tmp_path / "data")
        if hasattr(mod, "RAW"):
            monkeypatch.setattr(mod, "RAW", raw)
        if hasattr(mod, "PROCESSED"):
            monkeypatch.setattr(mod, "PROCESSED", processed)
        if hasattr(mod, "REPORTS"):
            monkeypatch.setattr(mod, "REPORTS", reports)
        if hasattr(mod, "FIGURES"):
            monkeypatch.setattr(mod, "FIGURES", figures)

    return tmp_path
