"""Train a reaction-center encoder with batched supervised contrastive loss."""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler

from catalyst_atlas.eval.labels import chemistry_label_col
from catalyst_atlas.eval.splits import make_splits
from catalyst_atlas.featurize.features import _aa_composition
from catalyst_atlas.featurize.graphs import graph_from_jsonable, permute_graph_node_features
from catalyst_atlas.models.device import get_device, require_torch
from catalyst_atlas.models.graph_encoder import ReactionCenterEncoder
from catalyst_atlas.models.pairs import sample_contrastive_batch
from catalyst_atlas.paths import PROCESSED, ROOT, ensure_dirs

logger = logging.getLogger(__name__)

ARTIFACTS = ROOT / "artifacts"


def _seed_everything(seed: int) -> None:
    """Seed numpy + torch so multi-seed bake-offs are meaningful."""
    torch = require_torch()
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _triplet_loss(za, zp, zn, margin: float = 0.2):
    """Legacy helper kept for unit tests."""
    torch = require_torch()
    d_pos = 1.0 - (za * zp).sum(dim=-1)
    d_neg = 1.0 - (za * zn).sum(dim=-1)
    return torch.relu(d_pos - d_neg + margin).mean()


def _supcon_loss(z_anchor, z_pos, z_negs, temperature: float = 0.1):
    """Legacy InfoNCE with one positive and multiple negatives (tests)."""
    torch = require_torch()
    pos = (z_anchor * z_pos).sum(dim=-1, keepdim=True) / temperature
    neg = torch.bmm(z_negs, z_anchor.unsqueeze(-1)).squeeze(-1) / temperature
    logits = torch.cat([pos, neg], dim=1)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return torch.nn.functional.cross_entropy(logits, labels)


def _batched_supcon_loss(
    z,
    chem_ids,
    mech_ids=None,
    temperature: float = 0.1,
    mech_boost: float = 0.15,
):
    """Multi-positive SupCon / NT-Xent over an in-batch similarity matrix.

    Same ``chemistry_family`` rows are positives; same ``mechanistic_pattern``
    positives get a small logit bump so mechanism-matched pairs pull harder.
    """
    torch = require_torch()
    b = z.size(0)
    if b < 2:
        return z.new_zeros(())

    # Clamp temperature away from zero for numerical stability.
    temperature = max(float(temperature), 1e-4)
    sim = (z @ z.T) / temperature
    eye = torch.eye(b, dtype=torch.bool, device=z.device)
    chem = chem_ids.view(-1, 1)
    pos_mask = (chem == chem.T) & ~eye

    # Optional mechanism weight bump on positive logits.
    if mech_ids is not None and mech_boost > 0:
        mech = mech_ids.view(-1, 1)
        same_mech = (mech == mech.T) & pos_mask
        sim = sim + float(mech_boost) * same_mech.to(sim.dtype)

    # Mask self with a large negative (avoid -inf → NaN in logsumexp edge cases).
    logits = sim.masked_fill(eye, -1e4)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

    pos_count = pos_mask.sum(dim=1).to(log_prob.dtype).clamp(min=1.0)
    mean_log = (log_prob * pos_mask.to(log_prob.dtype)).sum(dim=1) / pos_count
    has_pos = pos_mask.any(dim=1)
    if not bool(has_pos.any()):
        return z.new_zeros(())
    loss = -mean_log[has_pos].mean()
    if not torch.isfinite(loss):
        return z.new_zeros(())
    return loss


def _fold_val_split(
    meta: pd.DataFrame,
    train_idx: np.ndarray,
    n_val_folds: int = 12,
    seed: int = 7,
    min_val_enzymes: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out fold_clusters from the training split for fold-disconnected val."""
    rng = np.random.default_rng(seed)
    folds = meta.iloc[train_idx]["fold_cluster"].astype(str).unique().tolist()
    rng.shuffle(folds)
    n_hold = max(1, min(int(n_val_folds), max(1, len(folds) // 3)))
    # Grow hold-out until val has enough enzymes for stable kNN selection.
    val_folds: set[str] = set()
    for f in folds:
        val_folds.add(f)
        is_val = meta.iloc[train_idx]["fold_cluster"].astype(str).isin(val_folds).to_numpy()
        if int(is_val.sum()) >= min_val_enzymes and len(val_folds) >= n_hold:
            break
        if len(val_folds) >= max(n_hold, len(folds) // 3):
            break
    is_val = meta.iloc[train_idx]["fold_cluster"].astype(str).isin(val_folds).to_numpy()
    fit_idx = train_idx[~is_val]
    val_idx = train_idx[is_val]
    if len(fit_idx) == 0 or len(val_idx) < max(10, min_val_enzymes // 4):
        perm = rng.permutation(len(train_idx))
        n_val = max(min_val_enzymes, int(0.18 * len(train_idx)))
        n_val = min(n_val, len(train_idx) - 1)
        val_idx = train_idx[perm[:n_val]]
        fit_idx = train_idx[perm[n_val:]]
    return fit_idx, val_idx


def _knn_chemistry_accuracy(
    X_fit: np.ndarray,
    y_fit: list[str],
    X_val: np.ndarray,
    y_val: list[str],
    k: int = 5,
) -> float:
    if len(X_fit) == 0 or len(X_val) == 0:
        return 0.0
    kk = min(k, len(y_fit))
    nn = NearestNeighbors(n_neighbors=kk, metric="euclidean")
    nn.fit(X_fit)
    _, inds = nn.kneighbors(X_val)
    correct = 0
    for row, yt in zip(inds, y_val, strict=True):
        votes = [y_fit[i] for i in row]
        pred = max(set(votes), key=votes.count)
        correct += int(pred == yt)
    return correct / len(y_val)


def _convergent_mask(
    meta: pd.DataFrame,
    fit_idx: np.ndarray,
    val_idx: np.ndarray,
    label_col: str,
) -> np.ndarray:
    """Val enzymes whose chemistry appears in a different fold in fit set."""
    fit_chem_folds: dict[str, set] = {}
    for i in fit_idx:
        c = str(meta.iloc[int(i)][label_col])
        f = meta.iloc[int(i)]["fold_cluster"]
        fit_chem_folds.setdefault(c, set()).add(f)
    mask = []
    for i in val_idx:
        c = str(meta.iloc[int(i)][label_col])
        f = meta.iloc[int(i)]["fold_cluster"]
        folds = fit_chem_folds.get(c, set())
        mask.append(bool(folds) and f not in folds)
    return np.asarray(mask, dtype=bool)


def _encode_all(encoder, graphs, device, eng_matrix=None) -> np.ndarray:
    return encoder.encode_graphs(graphs, device=device, eng_matrix=eng_matrix)


def train_reaction_center_encoder(
    split: str = "fold_cluster",
    epochs: int = 250,
    batch_size: int = 32,
    lr: float = 3e-3,
    seed: int = 7,
    embed_dim: int = 64,
    hidden_dim: int = 64,
    test_size: float = 0.2,
    device: str | None = None,
    temperature: float = 0.1,
    lambda_cls: float = 0.3,
    patience: int = 50,
    n_val_folds: int = 12,
    fusion: bool = False,
    fusion_side: bool = False,
    fusion_esm: bool = False,
    random_graphs: bool = False,
    steps_per_epoch: int | None = None,
    min_epochs: int = 100,
    no_early_stop: bool = False,
    checkpoint_every: int = 10,
    val_ema: float = 0.7,
    conv_weight: float = 0.4,
) -> dict[str, Any]:
    """Train on one leakage-aware split's train set; encode full catalog."""
    torch = require_torch()
    ensure_dirs()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    _seed_everything(seed)

    n_modes = int(fusion) + int(fusion_side) + int(fusion_esm)
    if n_modes > 1:
        raise ValueError("Pass only one of --fusion / --fusion-side / --fusion-esm")
    if random_graphs and not fusion_esm:
        raise ValueError("--random-graphs is meant for --fusion-esm ablation")

    graphs_path = PROCESSED / "reaction_center_graphs.parquet"
    if not graphs_path.exists():
        from catalyst_atlas.featurize.graphs import build_graphs_table

        build_graphs_table()

    gdf = pd.read_parquet(graphs_path)
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

    if random_graphs:
        # Ablation: permute node features within each graph; keep topology.
        rng_g = np.random.default_rng(seed + 91)
        graphs = [permute_graph_node_features(g, rng_g) for g in graphs]
        logger.info("Random-graph ablation: shuffled node features on %d graphs", len(graphs))

    # Side vector for fusion readout (optional).
    # fusion → features_full; fusion_side → metal/cofactor; fusion_esm → frozen ESM-2.
    from catalyst_atlas.featurize.graphs import SIDE_DIM

    X_eng = None
    eng_dim = 0
    eng_scaler = None
    use_fusion = bool(fusion or fusion_side or fusion_esm)
    if fusion:
        feat_path = PROCESSED / "features_full.npy"
        if not feat_path.exists():
            raise FileNotFoundError(f"Missing {feat_path}; run cat-embed first")
        X_eng = np.load(feat_path).astype(np.float32)
        if len(X_eng) != len(meta):
            raise RuntimeError("features_full.npy length mismatch with meta")
        eng_dim = int(X_eng.shape[1])
    elif fusion_side:
        sides = []
        for g in graphs:
            s = g.get("side")
            if s is None:
                sides.append(np.zeros(SIDE_DIM, dtype=np.float32))
            else:
                sides.append(np.asarray(s, dtype=np.float32).reshape(-1))
        X_eng = np.stack(sides, axis=0)
        eng_dim = int(X_eng.shape[1])
    elif fusion_esm:
        esm_path = PROCESSED / "embedding_esm.npy"
        esm_meta_path = PROCESSED / "embedding_esm_meta.parquet"
        if not esm_path.exists():
            raise FileNotFoundError(
                "Missing embedding_esm.npy; run cat-esm before --fusion-esm"
            )
        X_eng = np.load(esm_path).astype(np.float32)
        if esm_meta_path.exists():
            esm_meta = pd.read_parquet(esm_meta_path).reset_index(drop=True)
            if not meta["enzyme_id"].astype(str).equals(esm_meta["enzyme_id"].astype(str)):
                emap = {str(e): i for i, e in enumerate(esm_meta["enzyme_id"])}
                X_eng = X_eng[[emap[str(e)] for e in meta["enzyme_id"]]]
        if len(X_eng) != len(meta):
            raise RuntimeError("embedding_esm.npy length mismatch with meta")
        eng_dim = int(X_eng.shape[1])
        logger.info("ESM+GNN fusion: frozen ESM side dim=%d", eng_dim)

    label_col = chemistry_label_col(meta)
    splits = make_splits(meta, test_size=test_size, seed=seed, label_col=label_col)
    if split not in splits:
        raise ValueError(f"Unknown split {split}; choose from {list(splits)}")
    train_idx, test_idx = splits[split]
    train_idx = np.asarray(train_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)

    fit_idx, val_idx = _fold_val_split(meta, train_idx, n_val_folds=n_val_folds, seed=seed)
    logger.info(
        "Fold-holdout val: fit=%d val=%d (held %d fold clusters from train)",
        len(fit_idx),
        len(val_idx),
        n_val_folds,
    )

    # Composition vectors for hard-negative mining (catalytic AA from graphs table).
    composition = np.stack(
        [_aa_composition(str(gdf.iloc[int(i)].get("catalytic_aas") or "")) for i in fit_idx]
    ).astype(np.float32)

    fit_meta = [
        {
            "chemistry_family": meta.iloc[i].get("chemistry_family"),
            "chemistry_class": meta.iloc[i].get("chemistry_class"),
            "mechanistic_pattern": meta.iloc[i].get("mechanistic_pattern"),
            "fold_cluster": meta.iloc[i].get("fold_cluster"),
            "ec_number": meta.iloc[i].get("ec_number"),
        }
        for i in fit_idx
    ]

    # Label encoders for aux classification (fit set only).
    chem_le = LabelEncoder()
    chem_labels_fit = [str(r["chemistry_family"] or r["chemistry_class"] or "unk") for r in fit_meta]
    chem_le.fit(chem_labels_fit)
    n_chem = len(chem_le.classes_)

    mech_raw = [str(r.get("mechanistic_pattern") or "unk") for r in fit_meta]
    mech_le = LabelEncoder()
    mech_le.fit(mech_raw)

    # Scale engineered features on fit set only.
    eng_fit = eng_val = eng_all = None
    if use_fusion and X_eng is not None:
        eng_scaler = StandardScaler()
        eng_fit = eng_scaler.fit_transform(X_eng[fit_idx]).astype(np.float32)
        eng_val = eng_scaler.transform(X_eng[val_idx]).astype(np.float32)
        eng_all = eng_scaler.transform(X_eng).astype(np.float32)

    dev = get_device() if device is None else torch.device(device)
    encoder = ReactionCenterEncoder(
        hidden_dim=hidden_dim, embed_dim=embed_dim, eng_dim=eng_dim
    ).to(dev)
    cls_head = torch.nn.Linear(embed_dim, n_chem).to(dev)
    params = list(encoder.parameters()) + list(cls_head.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    # Enough contrastive updates per epoch to see most of the fit set several times.
    n_steps = steps_per_epoch or max(40, (2 * len(fit_idx)) // max(1, batch_size))
    rng = np.random.default_rng(seed)

    logger.info(
        "Training reaction-center encoder on split=%s fit=%d val=%d "
        "steps/epoch=%d device=%s fusion=%s fusion_side=%s fusion_esm=%s "
        "no_early_stop=%s ckpt_every=%d lambda_cls=%.2f",
        split,
        len(fit_idx),
        len(val_idx),
        n_steps,
        dev,
        fusion,
        fusion_side,
        fusion_esm,
        no_early_stop,
        checkpoint_every,
        lambda_cls,
    )

    history: list[dict[str, float]] = []
    best_score = -1.0
    best_val_acc = -1.0
    best_state: dict[str, Any] | None = None
    best_epoch = -1
    stale = 0
    ema_score: float | None = None
    # Snapshot every N epochs for full bake-off (selected via val score).
    ckpt_every = max(1, int(checkpoint_every))
    snapshots: dict[int, dict[str, Any]] = {}

    label_col_y = label_col
    y_fit = meta.iloc[fit_idx][label_col_y].astype(str).tolist()
    y_val = meta.iloc[val_idx][label_col_y].astype(str).tolist()
    conv_mask = _convergent_mask(meta, fit_idx, val_idx, label_col_y)

    for epoch in range(epochs):
        encoder.train()
        cls_head.train()
        losses = []
        for _ in range(n_steps):
            local_batch = sample_contrastive_batch(
                fit_meta,
                rng,
                batch_size=batch_size,
                composition=composition,
            )
            if len(local_batch) < 2:
                continue
            opt.zero_grad()
            z_list = []
            chem_ids = []
            mech_ids = []
            for li in local_batch:
                gi = int(fit_idx[li])
                eng = None if eng_fit is None else eng_fit[li]
                z_list.append(encoder.encode_graph(graphs[gi], device=dev, eng=eng))
                chem_ids.append(
                    chem_le.transform(
                        [str(fit_meta[li]["chemistry_family"] or fit_meta[li]["chemistry_class"] or "unk")]
                    )[0]
                )
                mech_ids.append(
                    mech_le.transform([str(fit_meta[li].get("mechanistic_pattern") or "unk")])[0]
                )
            z = torch.stack(z_list)
            chem_t = torch.as_tensor(chem_ids, dtype=torch.long, device=dev)
            mech_t = torch.as_tensor(mech_ids, dtype=torch.long, device=dev)
            loss_sc = _batched_supcon_loss(z, chem_t, mech_ids=mech_t, temperature=temperature)
            logits = cls_head(z)
            loss_ce = torch.nn.functional.cross_entropy(logits, chem_t)
            loss = loss_sc + lambda_cls * loss_ce
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        mean_loss = float(np.mean(losses)) if losses else 0.0

        # Validation: encode fit+val, kNN chemistry accuracy.
        encoder.eval()
        with torch.no_grad():
            X_fit_emb = _encode_all(
                encoder,
                [graphs[int(i)] for i in fit_idx],
                device=dev,
                eng_matrix=eng_fit,
            )
            X_val_emb = _encode_all(
                encoder,
                [graphs[int(i)] for i in val_idx],
                device=dev,
                eng_matrix=eng_val,
            )
        val_acc = _knn_chemistry_accuracy(X_fit_emb, y_fit, X_val_emb, y_val, k=5)
        conv_acc = 0.0
        if conv_mask.any():
            conv_acc = _knn_chemistry_accuracy(
                X_fit_emb,
                y_fit,
                X_val_emb[conv_mask],
                [y for y, m in zip(y_val, conv_mask, strict=True) if m],
                k=5,
            )
        # Prefer checkpoints that also recover convergent chemistry.
        raw_score = (1.0 - conv_weight) * val_acc + conv_weight * conv_acc
        if ema_score is None:
            ema_score = raw_score
        else:
            ema_score = float(val_ema) * ema_score + (1.0 - float(val_ema)) * raw_score

        history.append(
            {
                "loss": mean_loss,
                "val_acc": val_acc,
                "val_convergent_acc": conv_acc,
                "val_score": raw_score,
                "val_score_ema": float(ema_score),
                "lr": float(scheduler.get_last_lr()[0]),
            }
        )
        improved = float(ema_score) > best_score + 1e-4
        if improved:
            best_score = float(ema_score)
            best_val_acc = val_acc
            best_epoch = epoch
            stale = 0
            best_state = {
                "encoder": {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()},
                "cls_head": {k: v.detach().cpu().clone() for k, v in cls_head.state_dict().items()},
            }
        else:
            stale += 1

        ep = epoch + 1
        if ep % ckpt_every == 0 or ep == epochs:
            snapshots[ep] = {
                "encoder": {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()},
                "cls_head": {k: v.detach().cpu().clone() for k, v in cls_head.state_dict().items()},
                "val_acc": val_acc,
                "val_convergent_acc": conv_acc,
                "val_score_ema": float(ema_score),
            }

        if epoch % 5 == 0 or epoch == epochs - 1 or improved:
            logger.info(
                "epoch %d/%d loss=%.4f val_acc=%.3f conv=%.3f ema=%.3f best=%.3f@%d lr=%.2e",
                ep,
                epochs,
                mean_loss,
                val_acc,
                conv_acc,
                float(ema_score),
                best_score,
                best_epoch + 1,
                scheduler.get_last_lr()[0],
            )
        if (not no_early_stop) and ep >= min_epochs and stale >= patience:
            logger.info(
                "Early stop at epoch %d (patience=%d, min_epochs=%d)",
                ep,
                patience,
                min_epochs,
            )
            break

    # Keep final weights as a candidate; choose selected vs final vs snapshots on val EMA/score.
    final_state = {
        "encoder": {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()},
        "cls_head": {k: v.detach().cpu().clone() for k, v in cls_head.state_dict().items()},
    }
    candidates: list[tuple[str, dict[str, Any], float]] = []
    if best_state is not None:
        candidates.append(("ema_best", best_state, best_score))
    candidates.append(("final", final_state, float(history[-1]["val_score_ema"]) if history else -1.0))
    for ep, snap in snapshots.items():
        candidates.append(
            (f"epoch_{ep}", {"encoder": snap["encoder"], "cls_head": snap["cls_head"]}, float(snap["val_score_ema"]))
        )
    # Re-score every candidate on current val set (fresh encode) for a fair bake-off.
    bakeoff: list[dict[str, Any]] = []
    pick_name, pick_state, pick_score = candidates[0]
    for name, state, _prior in candidates:
        encoder.load_state_dict(state["encoder"])
        encoder.eval()
        with torch.no_grad():
            X_fit_emb = _encode_all(
                encoder, [graphs[int(i)] for i in fit_idx], device=dev, eng_matrix=eng_fit
            )
            X_val_emb = _encode_all(
                encoder, [graphs[int(i)] for i in val_idx], device=dev, eng_matrix=eng_val
            )
        va = _knn_chemistry_accuracy(X_fit_emb, y_fit, X_val_emb, y_val, k=5)
        ca = 0.0
        if conv_mask.any():
            ca = _knn_chemistry_accuracy(
                X_fit_emb,
                y_fit,
                X_val_emb[conv_mask],
                [y for y, m in zip(y_val, conv_mask, strict=True) if m],
                k=5,
            )
        score = (1.0 - conv_weight) * va + conv_weight * ca
        bakeoff.append({"name": name, "val_acc": va, "val_convergent_acc": ca, "score": score})
        if score > pick_score + 1e-6:
            pick_name, pick_state, pick_score = name, state, score

    encoder.load_state_dict(pick_state["encoder"])
    cls_head.load_state_dict(pick_state["cls_head"])
    logger.info(
        "Selected checkpoint=%s score=%.3f (ema_best_epoch=%d bakeoff=%s)",
        pick_name,
        pick_score,
        best_epoch + 1,
        bakeoff,
    )

    encoder.eval()
    embeddings = _encode_all(encoder, graphs, device=dev, eng_matrix=eng_all)

    if fusion_esm and random_graphs:
        emb_path = PROCESSED / "embedding_esm_gnn_randnodes.npy"
        meta_out = PROCESSED / "embedding_esm_gnn_randnodes_meta.parquet"
        ckpt_path = ARTIFACTS / "reaction_center_esm_gnn_randnodes.pt"
        summary_name = "train_esm_gnn_randnodes_summary.json"
    elif fusion_esm:
        emb_path = PROCESSED / "embedding_esm_gnn.npy"
        meta_out = PROCESSED / "embedding_esm_gnn_meta.parquet"
        ckpt_path = ARTIFACTS / "reaction_center_esm_gnn.pt"
        summary_name = "train_esm_gnn_summary.json"
    elif use_fusion:
        emb_path = PROCESSED / "embedding_fusion.npy"
        meta_out = PROCESSED / "embedding_fusion_meta.parquet"
        ckpt_path = ARTIFACTS / "reaction_center_fusion.pt"
        summary_name = "train_fusion_summary.json"
    else:
        emb_path = PROCESSED / "embedding_learned.npy"
        meta_out = PROCESSED / "embedding_learned_meta.parquet"
        ckpt_path = ARTIFACTS / "reaction_center_encoder.pt"
        summary_name = "train_encoder_summary.json"
    np.save(emb_path, embeddings)
    meta.copy().to_parquet(meta_out, index=False)

    ckpt = {
        "state_dict": encoder.state_dict(),
        "cls_head": cls_head.state_dict(),
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "eng_dim": eng_dim,
        "fusion": fusion,
        "fusion_side": fusion_side,
        "fusion_esm": fusion_esm,
        "random_graphs": random_graphs,
        "split": split,
        "seed": seed,
        "epochs_ran": len(history),
        "best_epoch": best_epoch + 1,
        "best_val_acc": best_val_acc,
        "selected_checkpoint": pick_name,
        "selected_score": pick_score,
        "bakeoff": bakeoff,
        "history": history,
        "n_train": int(len(train_idx)),
        "n_fit": int(len(fit_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "lambda_cls": lambda_cls,
        "temperature": temperature,
        "no_early_stop": no_early_stop,
        "chem_classes": list(chem_le.classes_),
    }
    torch.save(ckpt, ckpt_path)

    summary = {
        "n_enzymes": int(len(embeddings)),
        "embed_dim": embed_dim,
        "split": split,
        "fusion": fusion,
        "fusion_side": fusion_side,
        "fusion_esm": fusion_esm,
        "random_graphs": random_graphs,
        "final_loss": history[-1]["loss"] if history else None,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch + 1,
        "selected_checkpoint": pick_name,
        "selected_score": pick_score,
        "bakeoff": bakeoff,
        "embedding_path": str(emb_path),
        "checkpoint": str(ckpt_path),
    }
    (ARTIFACTS / summary_name).write_text(json.dumps(summary, indent=2))
    logger.info("Wrote %s and %s", emb_path, ckpt_path)
    return summary
