"""Unit tests for random-graph ablation helper."""

from __future__ import annotations

import numpy as np

from catalyst_atlas.featurize.graphs import permute_graph_node_features


def test_permute_keeps_topology():
    rng = np.random.default_rng(0)
    g = {
        "x": np.arange(12, dtype=np.float32).reshape(3, 4),
        "edge_index": np.array([[0, 1], [1, 2]], dtype=np.int64).T,
        "edge_attr": np.ones((2, 2), dtype=np.float32),
        "n_nodes": 3,
        "n_edges": 2,
        "node_meta": [{"aa": "H"}, {"aa": "E"}, {"aa": "D"}],
        "side": np.zeros(5, dtype=np.float32),
    }
    out = permute_graph_node_features(g, rng)
    assert out["edge_index"].tolist() == g["edge_index"].tolist()
    assert out["x"].shape == g["x"].shape
    # Rows are a permutation of the original.
    orig = {tuple(r) for r in g["x"]}
    new = {tuple(r) for r in out["x"]}
    assert orig == new
    assert len(out["node_meta"]) == 3
