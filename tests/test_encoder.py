"""Learned encoder smoke tests — skipped when torch is not installed."""

from __future__ import annotations

import numpy as np
import pytest


def _tiny_graph(seed: int = 0):
    from catalyst_atlas.featurize.graphs import build_reaction_center_graph

    rng = np.random.default_rng(seed)
    micro = {
        "catalytic": [
            {"aa": "S", "resnum": 1, "xyz": rng.normal(size=3).tolist()},
            {"aa": "H", "resnum": 2, "xyz": (rng.normal(size=3) + 3).tolist()},
            {"aa": "D", "resnum": 3, "xyz": (rng.normal(size=3) + [0, 3, 0]).tolist()},
        ],
        "first_shell": [],
        "ligands": [],
    }
    return build_reaction_center_graph(micro)


def test_encoder_forward_and_normalize():
    torch = pytest.importorskip("torch")
    from catalyst_atlas.models.graph_encoder import ReactionCenterEncoder

    enc = ReactionCenterEncoder(hidden_dim=32, embed_dim=16, n_layers=2)
    enc.eval()
    g = _tiny_graph()
    z = enc.encode_graph(g)
    assert z.shape == (16,)
    assert torch.allclose(z.norm(), torch.tensor(1.0), atol=1e-4)


def test_encoder_batch_encode():
    pytest.importorskip("torch")
    from catalyst_atlas.models.graph_encoder import ReactionCenterEncoder

    enc = ReactionCenterEncoder(hidden_dim=32, embed_dim=16, n_layers=2)
    graphs = [_tiny_graph(i) for i in range(3)]
    X = enc.encode_graphs(graphs)
    assert X.shape == (3, 16)
    norms = np.linalg.norm(X, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-4)


def test_train_step_runs():
    torch = pytest.importorskip("torch")
    from catalyst_atlas.models.graph_encoder import ReactionCenterEncoder
    from catalyst_atlas.models.pairs import mine_indices, sample_contrastive_batch
    from catalyst_atlas.models.train_encoder import (
        _batched_supcon_loss,
        _supcon_loss,
        _triplet_loss,
    )

    enc = ReactionCenterEncoder(hidden_dim=32, embed_dim=16, n_layers=2)
    opt = torch.optim.Adam(enc.parameters(), lr=1e-2)
    graphs = [_tiny_graph(i) for i in range(6)]
    meta = [
        {
            "chemistry_family": "a",
            "fold_cluster": 0,
            "ec_number": "3.1",
            "mechanistic_pattern": "m",
        },
        {
            "chemistry_family": "a",
            "fold_cluster": 1,
            "ec_number": "3.1",
            "mechanistic_pattern": "m",
        },
        {
            "chemistry_family": "b",
            "fold_cluster": 0,
            "ec_number": "1.1",
            "mechanistic_pattern": "n",
        },
        {
            "chemistry_family": "b",
            "fold_cluster": 2,
            "ec_number": "1.1",
            "mechanistic_pattern": "n",
        },
        {
            "chemistry_family": "a",
            "fold_cluster": 3,
            "ec_number": "3.1",
            "mechanistic_pattern": "m",
        },
        {
            "chemistry_family": "b",
            "fold_cluster": 4,
            "ec_number": "1.1",
            "mechanistic_pattern": "n",
        },
    ]
    trips = mine_indices(meta, np.random.default_rng(0), n_pos_per_anchor=1, n_neg_per_anchor=2)
    assert trips

    enc.train()
    a, p, n = trips[0]
    opt.zero_grad()
    za = enc.encode_graph(graphs[a])
    zp = enc.encode_graph(graphs[p])
    zn = enc.encode_graph(graphs[n])
    loss = _supcon_loss(
        za.unsqueeze(0), zp.unsqueeze(0), zn.unsqueeze(0).unsqueeze(1)
    ) + _triplet_loss(za.unsqueeze(0), zp.unsqueeze(0), zn.unsqueeze(0))
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)

    # Batched multi-positive SupCon path (fresh encoder — avoid dirty grads).
    enc2 = ReactionCenterEncoder(hidden_dim=32, embed_dim=16, n_layers=2)
    batch = sample_contrastive_batch(meta, np.random.default_rng(1), batch_size=6)
    assert len(batch) >= 2
    chem_map = {"a": 0, "b": 1}
    z = torch.stack([enc2.encode_graph(graphs[i]) for i in batch])
    chem_ids = torch.tensor([chem_map[meta[i]["chemistry_family"]] for i in batch])
    loss2 = _batched_supcon_loss(z, chem_ids)
    assert torch.isfinite(loss2)
    loss2.backward()


def test_fusion_encoder_eng_dim():
    torch = pytest.importorskip("torch")
    from catalyst_atlas.models.graph_encoder import ReactionCenterEncoder

    eng_dim = 8
    enc = ReactionCenterEncoder(hidden_dim=32, embed_dim=16, n_layers=2, eng_dim=eng_dim)
    g = _tiny_graph()
    eng = np.random.default_rng(0).normal(size=eng_dim).astype(np.float32)
    z = enc.encode_graph(g, eng=eng)
    assert z.shape == (16,)
    assert torch.allclose(z.norm(), torch.tensor(1.0), atol=1e-4)
