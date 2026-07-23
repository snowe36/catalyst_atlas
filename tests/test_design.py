"""Unit tests for generative redesign (no GPU / ProteinMPNN / AF2)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from catalyst_atlas.data.generate_demo import generate_demo_atlas, save_raw_atlas
from catalyst_atlas.design.funnel import fixed_backbone_chemistry, run_funnel
from catalyst_atlas.design.generate import (
    DesignInvariantError,
    assert_design_invariants,
    generate_mock_designs,
    mutate_shell,
    run_generate,
)
from catalyst_atlas.design.mpnn import (
    fixed_positions_from_pocket,
    parse_design_fasta,
    write_design_fasta,
)
from catalyst_atlas.design.panel import passes_pocket_qc, resolve_panel
from catalyst_atlas.design.pocket import build_pocket, run_pockets
from catalyst_atlas.design.report import write_design_case_study
from catalyst_atlas.design.score import (
    chemistry_preservation_score,
    geometry_preservation,
    reference_geometry_vector,
    run_score,
)
from catalyst_atlas.site.extract import FIRST_SHELL_RADIUS


def _toy_atlas_row(**overrides):
    catalytic = [
        {"chain": "A", "resnum": 10, "aa": "H", "role": "catalytic", "xyz": [0.0, 0.0, 0.0]},
        {"chain": "A", "resnum": 35, "aa": "D", "role": "catalytic", "xyz": [3.5, 0.2, 0.1]},
        {"chain": "A", "resnum": 60, "aa": "E", "role": "catalytic", "xyz": [0.3, 3.8, -0.2]},
    ]
    shell = []
    for j in range(8):
        shell.append(
            {
                "chain": "A",
                "resnum": 100 + j,
                "aa": "N",
                "role": "first_shell",
                "xyz": [1.0 + 0.3 * j, 1.0, 1.0],
            }
        )
    for j in range(4):
        shell.append(
            {
                "chain": "A",
                "resnum": 150 + j,
                "aa": "V",
                "role": "second_shell",
                # ~10–11 Å from catalytic centroid → second shell
                "xyz": [0.0, 12.0 + 0.3 * j, 0.0],
            }
        )
    seq = ["A"] * 200
    for r in catalytic + shell:
        seq[r["resnum"] - 1] = r["aa"]
    row = {
        "enzyme_id": "TOYDES01",
        "pdb_id": "TOY1",
        "uniprot_id": "TOY",
        "family_id": "toy_hydrolysis",
        "enzyme_name": "toy hydrolase",
        "chemistry_family": "hydrolysis",
        "mechanistic_pattern": "metal activation",
        "chemistry_class": "hydrolysis",
        "catalytic_pattern": "His-Asp-Glu",
        "cofactor_tags": "Zn",
        "ec_number": "3.4.24.1",
        "sequence": "".join(seq),
        "site_residues_json": json.dumps(catalytic + shell),
        "ligands_json": json.dumps(
            [{"name": "Zn", "kind": "metal", "xyz": [1.2, 1.0, 0.5]}]
        ),
    }
    row.update(overrides)
    return pd.Series(row)


def test_build_pocket_shells_and_metadata():
    pocket = build_pocket(_toy_atlas_row())
    assert pocket["n_catalytic"] == 3
    assert pocket["n_first_shell"] >= 1
    assert pocket["n_second_shell"] >= 1
    assert all("xyz" in r and "aa" in r and "resnum" in r for r in pocket["catalytic_residues"])
    assert all("shell" in r and "seq_index" in r for r in pocket["redesignable"])
    for r in pocket["redesignable"]:
        d = r["dist_to_core"]
        if r["shell"] == "first":
            assert d <= FIRST_SHELL_RADIUS
        else:
            assert d > FIRST_SHELL_RADIUS
    assert pocket["catalytic_residues"][0]["seq_index"] == 9


def test_design_invariants_accept_shell_only():
    pocket = build_pocket(_toy_atlas_row())
    wt = pocket["sequence"]
    designed = mutate_shell(wt, pocket, n_mutations=3, rng=np.random.default_rng(0))
    assert_design_invariants(designed, wt, pocket)
    assert designed != wt


def test_design_invariants_reject_catalytic_change():
    pocket = build_pocket(_toy_atlas_row())
    wt = pocket["sequence"]
    bad = list(wt)
    idx = pocket["catalytic_residues"][0]["seq_index"]
    bad[idx] = "A" if bad[idx] != "A" else "G"
    with pytest.raises(DesignInvariantError):
        assert_design_invariants("".join(bad), wt, pocket)


def test_design_invariants_reject_outside_shell():
    pocket = build_pocket(_toy_atlas_row())
    wt = pocket["sequence"]
    redesignable = {r["seq_index"] for r in pocket["redesignable"]}
    outside = next(i for i in range(len(wt)) if i not in redesignable and i not in {
        r["seq_index"] for r in pocket["catalytic_residues"]
    })
    bad = list(wt)
    bad[outside] = "W" if bad[outside] != "W" else "Y"
    with pytest.raises(DesignInvariantError):
        assert_design_invariants("".join(bad), wt, pocket)


def test_fixed_positions_are_1based_catalytic():
    pocket = build_pocket(_toy_atlas_row())
    fixed = fixed_positions_from_pocket(pocket)
    assert fixed["fixed_positions"]["A"] == sorted(
        r["seq_index"] + 1 for r in pocket["catalytic_residues"]
    )


def test_chemistry_preservation_score_weights():
    s = chemistry_preservation_score(geometry=1.0, structure=0.0, esm=0.0)
    assert abs(s - 0.4) < 1e-9
    assert geometry_preservation(
        reference_geometry_vector(build_pocket(_toy_atlas_row())),
        reference_geometry_vector(build_pocket(_toy_atlas_row())),
    ) == pytest.approx(1.0)


def test_fasta_roundtrip(tmp_path):
    records = [
        {"enzyme_id": "E1", "design_id": "d0", "sequence": "ACDE"},
        {"enzyme_id": "E1", "design_id": "d1", "sequence": "ACDF"},
    ]
    path = tmp_path / "d.fasta"
    write_design_fasta(records, path)
    parsed = parse_design_fasta(path)
    assert len(parsed) == 2
    assert parsed[0]["sequence"] == "ACDE"


def test_end_to_end_mock_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr("catalyst_atlas.paths.RAW", tmp_path / "raw")
    monkeypatch.setattr("catalyst_atlas.paths.PROCESSED", tmp_path / "processed")
    monkeypatch.setattr("catalyst_atlas.paths.REPORTS", tmp_path / "reports")
    monkeypatch.setattr("catalyst_atlas.paths.FIGURES", tmp_path / "reports" / "figures")
    monkeypatch.setattr("catalyst_atlas.design.pocket.RAW", tmp_path / "raw")
    monkeypatch.setattr("catalyst_atlas.design.pocket.PROCESSED", tmp_path / "processed")
    monkeypatch.setattr("catalyst_atlas.design.panel.RAW", tmp_path / "raw")
    monkeypatch.setattr("catalyst_atlas.design.panel.PROCESSED", tmp_path / "processed")
    monkeypatch.setattr("catalyst_atlas.design.generate.PROCESSED", tmp_path / "processed")
    monkeypatch.setattr("catalyst_atlas.design.mpnn.PROCESSED", tmp_path / "processed")
    monkeypatch.setattr("catalyst_atlas.design.predict.PROCESSED", tmp_path / "processed")
    monkeypatch.setattr("catalyst_atlas.design.score.PROCESSED", tmp_path / "processed")
    monkeypatch.setattr("catalyst_atlas.design.report.PROCESSED", tmp_path / "processed")
    monkeypatch.setattr("catalyst_atlas.design.report.REPORTS", tmp_path / "reports")
    monkeypatch.setattr("catalyst_atlas.design.report.FIGURES", tmp_path / "reports" / "figures")
    monkeypatch.setattr("catalyst_atlas.data.generate_demo.RAW", tmp_path / "raw")

    (tmp_path / "raw").mkdir(parents=True)
    (tmp_path / "processed").mkdir(parents=True)
    df = generate_demo_atlas(n_enzymes=60, seed=7)
    save_raw_atlas(df)

    panel = resolve_panel(df, target_size=5)
    assert 1 <= len(panel) <= 5
    eids = [p["enzyme_id"] for p in panel]
    for eid in eids:
        assert passes_pocket_qc(build_pocket(df[df["enzyme_id"] == eid].iloc[0]))

    run_pockets(enzyme_ids=eids)
    designs = run_generate(eids, n_sequences=8, use_mock=True, seed=7)
    assert len(designs) == 8 * len(eids)
    scores = run_score(eids, mock_predictions=True, seed=7)
    assert (scores["is_wt"]).sum() == len(eids)
    assert "chemistry_preservation_score" in scores.columns
    assert (scores.loc[scores["is_wt"], "delta_score_vs_wt"] == 0).all()

    report = write_design_case_study(scores, panel=panel)
    assert report.exists()
    assert "chemistry_preservation_score" in report.read_text()


def test_mock_designs_respect_invariants():
    pocket = build_pocket(_toy_atlas_row())
    records = generate_mock_designs(pocket, n_sequences=20, seed=3)
    wt = pocket["sequence"]
    for rec in records:
        assert_design_invariants(rec["sequence"], wt, pocket)


def test_funnel_shortlists_top_k(tmp_path, monkeypatch):
    monkeypatch.setattr("catalyst_atlas.design.funnel.PROCESSED", tmp_path / "processed")
    monkeypatch.setattr("catalyst_atlas.design.mpnn.PROCESSED", tmp_path / "processed")
    monkeypatch.setattr("catalyst_atlas.design.pocket.PROCESSED", tmp_path / "processed")
    (tmp_path / "processed" / "design" / "pockets").mkdir(parents=True)

    pocket = build_pocket(_toy_atlas_row())
    (tmp_path / "processed" / "design" / "pockets" / f"{pocket['enzyme_id']}.json").write_text(
        json.dumps(pocket)
    )
    records = generate_mock_designs(pocket, n_sequences=30, seed=1)
    designs = pd.DataFrame(records)
    designs["n_mutations"] = [
        sum(a != b for a, b in zip(pocket["sequence"], s, strict=True))
        for s in designs["sequence"]
    ]
    designs["mutations"] = ""
    designs.to_parquet(tmp_path / "processed" / "design" / "designs.parquet", index=False)

    chem = fixed_backbone_chemistry(designs.iloc[0]["sequence"], pocket["sequence"], pocket)
    assert 0.0 <= chem <= 1.0

    meta = run_funnel(designs, top_k=5, enzyme_ids=[pocket["enzyme_id"]])
    assert meta["n_input_designs"] == 30
    assert meta["n_af_designs"] == 5
    assert meta["n_af_wt"] == 1
    assert meta["n_af_total"] == 6

