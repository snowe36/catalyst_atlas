"""Eval includes ESM / learned methods only when embedding artifacts exist."""

from __future__ import annotations

import numpy as np
import pandas as pd

from catalyst_atlas.eval.baselines import knn_transfer
from catalyst_atlas.eval.run import _align_embedding, _method_predictions
from catalyst_atlas.paths import PROCESSED


def test_align_embedding_orders_by_enzyme_id(tmp_path, monkeypatch):
    meta = pd.DataFrame({"enzyme_id": ["A", "B", "C"]})
    emb_meta = pd.DataFrame({"enzyme_id": ["C", "A", "B"]})
    X = np.array([[3.0], [1.0], [2.0]], dtype=np.float32)
    emb_path = tmp_path / "embedding_x.npy"
    meta_path = tmp_path / "embedding_x_meta.parquet"
    np.save(emb_path, X)
    emb_meta.to_parquet(meta_path, index=False)
    aligned = _align_embedding(meta, emb_path, meta_path)
    assert aligned is not None
    np.testing.assert_array_equal(aligned.ravel(), [1.0, 2.0, 3.0])


def test_method_predictions_includes_optional_tracks():
    # Tiny synthetic matrices — no disk artifacts required.
    meta = pd.DataFrame(
        {
            "enzyme_id": [f"E{i}" for i in range(8)],
            "chemistry_family": ["hydrolysis", "hydrolysis", "oxidation", "oxidation"] * 2,
            "chemistry_class": ["hydrolysis", "hydrolysis", "oxidation", "oxidation"] * 2,
            "seq_cluster": [0, 0, 1, 1, 2, 2, 3, 3],
            "fold_cluster": [0, 1, 0, 1, 2, 3, 2, 3],
            "sequence": ["AAAA", "AAAC", "GGGG", "GGGA", "TTTT", "TTTA", "CCCC", "CCCA"],
            "is_cryptic_seed": [False] * 8,
        }
    )
    rng = np.random.default_rng(0)
    X_full = rng.normal(size=(8, 6)).astype(np.float32)
    X_comp = rng.normal(size=(8, 4)).astype(np.float32)
    X_esm = rng.normal(size=(8, 8)).astype(np.float32)
    # L2-normalize fake learned embeddings
    X_learned = rng.normal(size=(8, 8)).astype(np.float32)
    X_learned /= np.linalg.norm(X_learned, axis=1, keepdims=True) + 1e-8

    train_idx = np.array([0, 1, 2, 3])
    test_idx = np.array([4, 5, 6, 7])
    _, methods, _, _ = _method_predictions(
        meta,
        X_full,
        X_comp,
        train_idx,
        test_idx,
        k=2,
        label_col="chemistry_family",
        X_esm=X_esm,
        X_learned=X_learned,
    )
    assert "esm2_transfer" in methods
    assert "learned_catalytic_encoder" in methods
    assert "catalyst_microenvironment" in methods
    assert len(methods["esm2_transfer"]) == 4
    assert len(methods["learned_catalytic_encoder"]) == 4


def test_knn_transfer_smoke():
    X_tr = np.eye(4, dtype=np.float32)
    y = ["a", "a", "b", "b"]
    X_te = np.array([[1.0, 0, 0, 0], [0, 0, 1.0, 0]], dtype=np.float32)
    preds = knn_transfer(X_tr, y, X_te, k=1)
    assert preds == ["a", "b"]


def test_processed_graphs_artifact_exists_or_skip():
    # Integration guard when local processed data is present (dev machines).
    path = PROCESSED / "reaction_center_graphs.parquet"
    if not path.exists():
        return
    df = pd.read_parquet(path)
    assert len(df) > 0
    assert {"enzyme_id", "graph_json", "n_nodes", "n_edges"} <= set(df.columns)
