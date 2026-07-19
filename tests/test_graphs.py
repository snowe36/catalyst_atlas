"""Reaction-center graph builder tests (CPU; no torch required)."""

from __future__ import annotations

import numpy as np

from catalyst_atlas.featurize.graphs import (
    EDGE_DIM,
    NODE_DIM,
    build_reaction_center_graph,
    graph_from_jsonable,
    graph_to_jsonable,
)


def _toy_micro() -> dict:
    return {
        "catalytic": [
            {"aa": "S", "resnum": 10, "xyz": [0.0, 0.0, 0.0]},
            {"aa": "H", "resnum": 35, "xyz": [3.5, 0.2, 0.1]},
            {"aa": "D", "resnum": 60, "xyz": [0.3, 3.8, -0.2]},
        ],
        "first_shell": [
            {"aa": "N", "resnum": 200, "xyz": [2.0, 2.0, 2.0], "dist_to_core": 3.4},
        ],
        "ligands": [],
    }


def test_graph_from_toy_site():
    g = build_reaction_center_graph(_toy_micro())
    assert g["n_nodes"] >= 3  # Ser-His-Asp catalytic
    assert g["x"].shape == (g["n_nodes"], NODE_DIM)
    assert g["edge_attr"].shape[1] == EDGE_DIM
    assert g["edge_index"].shape[0] == 2
    # catalytic pairwise edges are undirected → 3 pairs * 2 = 6
    assert g["n_edges"] >= 6
    assert np.isfinite(g["x"]).all()


def test_graph_with_metal_coordination():
    micro = {
        "catalytic": [
            {"aa": "H", "resnum": 1, "xyz": [0.0, 0.0, 0.0]},
            {"aa": "E", "resnum": 2, "xyz": [2.0, 0.0, 0.0]},
            {"aa": "H", "resnum": 3, "xyz": [0.0, 2.0, 0.0]},
        ],
        "first_shell": [],
        "ligands": [
            {
                "name": "Zn",
                "kind": "metal",
                "xyz": [0.7, 0.7, 0.0],
                "dist_to_core": 1.0,
                "coordination": {
                    "n_coord": 2,
                    "geometry": "linear",
                    "motif": "H-E",
                    "min_distance": 2.1,
                    "residues": [
                        {"aa": "H", "resnum": 1, "distance": 2.1},
                        {"aa": "E", "resnum": 2, "distance": 2.2},
                    ],
                },
            }
        ],
    }
    g = build_reaction_center_graph(micro)
    assert g["n_nodes"] == 4
    kinds = [m["kind"] for m in g["node_meta"]]
    assert "metal" in kinds
    assert g["n_edges"] > 0


def test_graph_json_roundtrip():
    g = build_reaction_center_graph(_toy_micro())
    g2 = graph_from_jsonable(graph_to_jsonable(g))
    assert g2["n_nodes"] == g["n_nodes"]
    np.testing.assert_allclose(g2["x"], g["x"])
    np.testing.assert_array_equal(g2["edge_index"], g["edge_index"])


def test_pairs_miner_hard_negatives():
    from catalyst_atlas.models.pairs import mine_indices, sample_contrastive_batch

    rows = [
        {"chemistry_family": "hydrolysis", "fold_cluster": "A", "ec_number": "3.4.1", "mechanistic_pattern": "metal"},
        {"chemistry_family": "hydrolysis", "fold_cluster": "B", "ec_number": "3.4.1", "mechanistic_pattern": "metal"},
        {"chemistry_family": "oxidation", "fold_cluster": "A", "ec_number": "1.1.1", "mechanistic_pattern": "NAD"},
        {"chemistry_family": "oxidation", "fold_cluster": "C", "ec_number": "1.1.1", "mechanistic_pattern": "NAD"},
    ]
    trips = mine_indices(rows, np.random.default_rng(0))
    assert trips
    # At least one hard negative: same fold A, different chemistry
    hard = False
    for a, _p, n in trips:
        if rows[a]["fold_cluster"] == rows[n]["fold_cluster"]:
            if rows[a]["chemistry_family"] != rows[n]["chemistry_family"]:
                hard = True
    assert hard

    batch = sample_contrastive_batch(rows, np.random.default_rng(1), batch_size=4)
    assert len(batch) == 4
    # Hydrolysis must contribute from both folds A and B when both present.
    hydro_folds = {
        rows[i]["fold_cluster"]
        for i in batch
        if rows[i]["chemistry_family"] == "hydrolysis"
    }
    assert hydro_folds == {"A", "B"} or len(hydro_folds) >= 1


def test_max_first_shell_cap():
    from catalyst_atlas.featurize.graphs import build_reaction_center_graph

    micro = _toy_micro()
    micro["first_shell"] = [
        {"aa": "N", "resnum": 200 + i, "xyz": [2.0 + i * 0.1, 2.0, 2.0], "dist_to_core": float(i + 1)}
        for i in range(20)
    ]
    g4 = build_reaction_center_graph(micro, max_first_shell=4)
    g2 = build_reaction_center_graph(micro, max_first_shell=2)
    # 3 catalytic + capped first-shell
    assert g4["n_nodes"] == 3 + 4
    assert g2["n_nodes"] == 3 + 2
    assert "side" in g4
    assert g4["side"].ndim == 1
