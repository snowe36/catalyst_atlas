"""Annotation-style negative control audits."""

from __future__ import annotations

import numpy as np
import pandas as pd

from catalyst_atlas.data.uniprot_expand import attach_ec_labels, ec_labels
from catalyst_atlas.eval.diagnostics import annotation_style_audits


def test_ec_labels():
    assert ec_labels("3.4.21.1") == {"ec_class": "3", "ec3": "3.4.21"}
    assert ec_labels("")["ec_class"] == "unknown"


def test_attach_ec_labels():
    df = pd.DataFrame(
        {
            "enzyme_id": ["A", "B"],
            "ec_number": ["1.1.1.1", "3.4.21.4"],
            "source": ["mcsa", "mcsa"],
        }
    )
    out = attach_ec_labels(df)
    assert list(out["ec_class"]) == ["1", "3"]
    assert list(out["ec3"]) == ["1.1.1", "3.4.21"]
    assert "structure_source" in out.columns


def test_annotation_style_audits_smoke():
    rng = np.random.default_rng(0)
    n = 20
    meta = pd.DataFrame(
        {
            "enzyme_id": [f"E{i}" for i in range(n)],
            "chemistry_family": (["hydrolysis", "oxidation"] * (n // 2)),
            "catalytic_aas": (["HDE", "SER"] * (n // 2)),
            "cofactor_tags": (["Zn", "none", "NAD", "Zn"] * (n // 4)),
            "fold_cluster": list(range(n)),
        }
    )
    X = rng.normal(size=(n, 80)).astype(float)
    # Lay out fake shell blocks in the expected slices.
    X[:, 20:40] = rng.random((n, 20))
    X[:, 48:56] = rng.random((n, 8))
    train_idx = np.arange(0, 12)
    test_idx = np.arange(12, 20)
    y_train = meta.iloc[train_idx]["chemistry_family"].tolist()
    y_test = meta.iloc[test_idx]["chemistry_family"].tolist()
    preds = {
        "catalyst_microenvironment": y_test[:],  # perfect (for subset scoring)
        "esm2_transfer": ["hydrolysis"] * len(y_test),
    }
    out = annotation_style_audits(
        meta,
        train_idx,
        test_idx,
        preds,
        X_full=X,
        y_train=y_train,
        y_test=y_test,
        label_col="chemistry_family",
        k=3,
        seed=0,
    )
    assert "same_residues_different_chemistry" in out
    assert "same_cofactor_different_chemistry" in out
    assert "shuffled_first_shell" in out
    assert "decoy_reaction_centers" in out
    assert out["shuffled_first_shell"]["n"] == len(y_test)
    assert "catalyst_microenvironment_decoy" in out["decoy_reaction_centers"]["methods"]
