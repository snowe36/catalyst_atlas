"""Unit tests for identity stratification and fold/chemistry audits."""

from __future__ import annotations

import numpy as np
import pandas as pd

from catalyst_atlas.eval.diagnostics import (
    fold_chemistry_audits,
    identity_bin,
    nearest_train_sequence_identity,
    sequence_identity_stratified_transfer,
)


def test_identity_bin_edges():
    assert identity_bin(95) == ">80%"
    assert identity_bin(80) == ">80%"
    assert identity_bin(79.9) == "40–80%"
    assert identity_bin(40) == "40–80%"
    assert identity_bin(20) == "20–40%"
    assert identity_bin(19.9) == "<20%"
    assert identity_bin(0) == "<20%"


def test_nearest_train_identity_from_mmseqs():
    meta = pd.DataFrame(
        {
            "enzyme_id": ["A", "B", "C"],
            "sequence": ["M" * 40, "M" * 40, "A" * 40],
        }
    )
    hits = pd.DataFrame(
        {
            "query": ["C", "C"],
            "target": ["A", "B"],
            "pident": [35.0, 88.0],
            "score": [10.0, 50.0],
        }
    )
    train_idx = np.array([0, 1])
    test_idx = np.array([2])
    ident, src = nearest_train_sequence_identity(
        meta, train_idx, test_idx, mmseqs_hits=hits
    )
    assert src == "mmseqs_pident"
    assert ident[0] == 88.0


def test_sequence_identity_stratified_transfer():
    y_true = ["hydrolysis", "transfer", "transfer", "hydrolysis"]
    preds = {
        "catalyst_microenvironment": [
            "hydrolysis",
            "transfer",
            "hydrolysis",
            "hydrolysis",
        ],
        "mmseqs_transfer": ["hydrolysis", "transfer", "transfer", "transfer"],
    }
    nearest = np.array([90.0, 55.0, 25.0, 5.0])
    out = sequence_identity_stratified_transfer(y_true, preds, nearest)
    assert out["bin_counts"][">80%"] == 1
    assert out["bin_counts"]["40–80%"] == 1
    assert out["methods"]["catalyst_microenvironment"][">80%"] == 1.0
    assert out["methods"]["catalyst_microenvironment"]["20–40%"] == 0.0


def test_fold_chemistry_audits_trap_and_recovery():
    meta = pd.DataFrame(
        {
            "enzyme_id": ["t1", "t2", "q_trap", "q_conv"],
            "fold_cluster": [1, 2, 1, 9],
            "chemistry_family": [
                "hydrolysis",
                "transfer",
                "oxidation-reduction",
                "transfer",
            ],
        }
    )
    train_idx = np.array([0, 1])
    test_idx = np.array([2, 3])
    preds = {
        "catalyst_microenvironment": ["oxidation-reduction", "transfer"],
        "foldseek_transfer": ["hydrolysis", "__unseen__"],
        "fold_cluster_transfer": ["hydrolysis", "__unseen__"],
        "mmseqs_transfer": ["hydrolysis", "transfer"],
    }
    audits = fold_chemistry_audits(
        meta, train_idx, test_idx, preds, label_col="chemistry_family"
    )
    trap = audits["same_fold_different_chemistry"]
    conv = audits["different_fold_same_chemistry"]
    assert trap["n"] == 1
    assert conv["n"] == 1
    assert trap["methods"]["catalyst_microenvironment"]["accuracy"] == 1.0
    assert trap["methods"]["foldseek_transfer"]["accuracy"] == 0.0
    assert conv["methods"]["catalyst_microenvironment"]["accuracy"] == 1.0
