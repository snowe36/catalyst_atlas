"""Train a reaction-center encoder with supervised contrastive / triplet loss."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from catalyst_atlas.eval.labels import chemistry_label_col
from catalyst_atlas.eval.splits import make_splits
from catalyst_atlas.featurize.graphs import graph_from_jsonable
from catalyst_atlas.models.device import get_device, require_torch
from catalyst_atlas.models.graph_encoder import ReactionCenterEncoder
from catalyst_atlas.models.pairs import mine_indices
from catalyst_atlas.paths import PROCESSED, ROOT, ensure_dirs

logger = logging.getLogger(__name__)

ARTIFACTS = ROOT / "artifacts"


def _triplet_loss(za, zp, zn, margin: float = 0.2):
    torch = require_torch()
    # embeddings are L2-normalized → cosine distance = 1 - dot
    d_pos = 1.0 - (za * zp).sum(dim=-1)
    d_neg = 1.0 - (za * zn).sum(dim=-1)
    return torch.relu(d_pos - d_neg + margin).mean()


def _supcon_loss(z_anchor, z_pos, z_negs, temperature: float = 0.1):
    """InfoNCE with one positive and multiple negatives."""
    torch = require_torch()
    # z_negs: (B, N, D)
    pos = (z_anchor * z_pos).sum(dim=-1, keepdim=True) / temperature
    neg = torch.bmm(z_negs, z_anchor.unsqueeze(-1)).squeeze(-1) / temperature
    logits = torch.cat([pos, neg], dim=1)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return torch.nn.functional.cross_entropy(logits, labels)


def train_reaction_center_encoder(
    split: str = "fold_cluster",
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 7,
    embed_dim: int = 64,
    hidden_dim: int = 64,
    test_size: float = 0.2,
    device: str | None = None,
) -> dict[str, Any]:
    """Train on one leakage-aware split's train set; encode full catalog."""
    torch = require_torch()
    ensure_dirs()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    graphs_path = PROCESSED / "reaction_center_graphs.parquet"
    if not graphs_path.exists():
        from catalyst_atlas.featurize.graphs import build_graphs_table

        build_graphs_table()

    gdf = pd.read_parquet(graphs_path)
    # Align with feature meta for splits
    meta_path = PROCESSED / "features_full_meta.parquet"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}; run cat-embed first")
    meta = pd.read_parquet(meta_path).reset_index(drop=True)
    gmap = {eid: i for i, eid in enumerate(gdf["enzyme_id"].astype(str))}
    order = [gmap[str(eid)] for eid in meta["enzyme_id"]]
    gdf = gdf.iloc[order].reset_index(drop=True)

    graphs = [graph_from_jsonable(json.loads(s)) for s in gdf["graph_json"]]
    for i, row in gdf.iterrows():
        graphs[i]["enzyme_id"] = row["enzyme_id"]

    label_col = chemistry_label_col(meta)
    splits = make_splits(meta, test_size=test_size, seed=seed, label_col=label_col)
    if split not in splits:
        raise ValueError(f"Unknown split {split}; choose from {list(splits)}")
    train_idx, test_idx = splits[split]
    train_idx = np.asarray(train_idx, dtype=int)

    train_meta = [
        {
            "chemistry_family": meta.iloc[i].get("chemistry_family"),
            "chemistry_class": meta.iloc[i].get("chemistry_class"),
            "mechanistic_pattern": meta.iloc[i].get("mechanistic_pattern"),
            "fold_cluster": meta.iloc[i].get("fold_cluster"),
            "ec_number": meta.iloc[i].get("ec_number"),
        }
        for i in train_idx
    ]
    rng = np.random.default_rng(seed)
    triplets = mine_indices(train_meta, rng)
    if not triplets:
        raise RuntimeError("No training triplets mined; check label diversity")

    dev = get_device() if device is None else torch.device(device)
    encoder = ReactionCenterEncoder(hidden_dim=hidden_dim, embed_dim=embed_dim).to(dev)
    opt = torch.optim.Adam(encoder.parameters(), lr=lr)

    logger.info(
        "Training reaction-center encoder on split=%s train=%d triplets=%d device=%s",
        split,
        len(train_idx),
        len(triplets),
        dev,
    )

    history: list[float] = []
    for epoch in range(epochs):
        rng.shuffle(triplets)
        losses = []
        encoder.train()
        for start in range(0, len(triplets), batch_size):
            batch = triplets[start : start + batch_size]
            opt.zero_grad()
            za_list, zp_list, zn_list = [], [], []
            for a, p, n in batch:
                gi = int(train_idx[a])
                gj = int(train_idx[p])
                gk = int(train_idx[n])
                za_list.append(encoder.encode_graph(graphs[gi], device=dev))
                zp_list.append(encoder.encode_graph(graphs[gj], device=dev))
                zn_list.append(encoder.encode_graph(graphs[gk], device=dev))
            za = torch.stack(za_list)
            zp = torch.stack(zp_list)
            zn = torch.stack(zn_list)
            # one hard neg as N=1 InfoNCE + triplet auxiliary
            loss = _supcon_loss(za, zp, zn.unsqueeze(1)) + 0.5 * _triplet_loss(za, zp, zn)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        mean_loss = float(np.mean(losses)) if losses else 0.0
        history.append(mean_loss)
        if epoch % 5 == 0 or epoch == epochs - 1:
            logger.info("epoch %d/%d loss=%.4f", epoch + 1, epochs, mean_loss)

    encoder.eval()
    embeddings = encoder.encode_graphs(graphs, device=dev)
    emb_path = PROCESSED / "embedding_learned.npy"
    np.save(emb_path, embeddings)
    meta_out = meta.copy()
    meta_out.to_parquet(PROCESSED / "embedding_learned_meta.parquet", index=False)

    ckpt = {
        "state_dict": encoder.state_dict(),
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "split": split,
        "seed": seed,
        "epochs": epochs,
        "history": history,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
    }
    ckpt_path = ARTIFACTS / "reaction_center_encoder.pt"
    torch.save(ckpt, ckpt_path)

    summary = {
        "n_enzymes": int(len(embeddings)),
        "embed_dim": embed_dim,
        "split": split,
        "final_loss": history[-1] if history else None,
        "embedding_path": str(emb_path),
        "checkpoint": str(ckpt_path),
    }
    (ARTIFACTS / "train_encoder_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Wrote %s and %s", emb_path, ckpt_path)
    return summary


def run_train_encoder(**kwargs: Any) -> dict[str, Any]:
    return train_reaction_center_encoder(**kwargs)
