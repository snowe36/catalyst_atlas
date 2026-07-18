"""Unit tests for retrieval chemistry transfer (no mmseqs/foldseek required)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from catalyst_atlas.eval.external_baselines import (
    _normalize_hit_id,
    retrieval_chemistry_transfer,
    tool_status,
)


def test_tool_status_keys():
    status = tool_status()
    assert "mmseqs" in status
    assert "foldseek" in status


def test_normalize_hit_id_strips_foldseek_chain():
    assert _normalize_hit_id("MCSA00335_B") == "MCSA00335"
    assert _normalize_hit_id("MCSA00001.pdb") == "MCSA00001"
    assert _normalize_hit_id("MCSA00001") == "MCSA00001"


def test_retrieval_chemistry_transfer():
    meta = pd.DataFrame(
        {
            "enzyme_id": ["A", "B", "C", "D"],
            "chemistry_family": ["hydrolysis", "hydrolysis", "transfer", "transfer"],
        }
    )
    hits = pd.DataFrame(
        {
            "query": ["C", "C", "D"],
            "target": ["A", "B", "B"],
            "score": [10.0, 50.0, 5.0],
        }
    )
    train_idx = np.array([0, 1])
    test_idx = np.array([2, 3])
    preds = retrieval_chemistry_transfer(
        hits, meta, train_idx, test_idx, label_col="chemistry_family"
    )
    # C → best train hit B (score 50) → hydrolysis
    assert preds[0] == "hydrolysis"
    # D → best train hit B → hydrolysis
    assert preds[1] == "hydrolysis"
