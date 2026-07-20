#!/usr/bin/env python3
"""Multi-seed fold_cluster bake-off for ESM-2 vs ESM+GNN.

Protocol (seeds 7, 11, 13):
  train ESM+GNN --fusion-esm --seed S
  fold_cluster-only eval (same seed)
  record fold_cluster for esm2_transfer and esm_gnn_fusion

ESM2 variance is split sensitivity (embedding fixed).
ESM+GNN variance couples train init + split.

Env:
  V05_SKIP_TRAIN=1  — eval only with existing embedding_esm_gnn.npy (smoke)
  V05_RESUME=1      — skip train when embedding_esm_gnn_seed{S}.npy exists (default)
  V05_SEEDS=7,11,13
  V05_EPOCHS=200
"""

from __future__ import annotations

import json
import os
import shutil
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from catalyst_atlas.paths import PROCESSED, REPORTS, ROOT, ensure_dirs  # noqa: E402

ARTIFACTS = ROOT / "artifacts"


def _log(msg: str) -> None:
    print(msg, flush=True)


def _mean_std(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    if len(vals) == 1:
        return {"mean": vals[0], "std": 0.0, "n": 1}
    return {
        "mean": float(statistics.mean(vals)),
        "std": float(statistics.stdev(vals)),
        "n": len(vals),
    }


def _fold_cluster_scores(seed: int, k: int = 5) -> dict[str, float | None]:
    """Lightweight fold_cluster eval — no k-mer matrix, diagnostics, or plots."""
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    from catalyst_atlas.data.labels import chemistry_label_col
    from catalyst_atlas.eval.baselines import knn_transfer
    from catalyst_atlas.eval.run import _align_embedding, _load_unscaled_features
    from catalyst_atlas.eval.splits import make_splits

    meta, X_full, _X_comp = _load_unscaled_features()
    label_col = chemistry_label_col(meta)
    X_esm = _align_embedding(
        meta, PROCESSED / "embedding_esm.npy", PROCESSED / "embedding_esm_meta.parquet"
    )
    X_gnn = _align_embedding(
        meta,
        PROCESSED / "embedding_esm_gnn.npy",
        PROCESSED / "embedding_esm_gnn_meta.parquet",
    )
    if X_esm is None or X_gnn is None:
        raise RuntimeError("Need embedding_esm.npy and embedding_esm_gnn.npy for bake-off eval")

    splits = make_splits(meta, test_size=0.2, seed=seed, label_col=label_col)
    train_idx, test_idx = splits["fold_cluster"]
    y_train = meta.iloc[train_idx][label_col].astype(str).tolist()
    y_test = meta.iloc[test_idx][label_col].astype(str).tolist()

    def _scaled_knn(X: np.ndarray) -> float:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        preds = knn_transfer(X_tr, y_train, X_te, k=k)
        return float(np.mean([p == t for p, t in zip(preds, y_test, strict=True)]))

    # ESM+GNN is already L2 chemistry space — match run_eval (no re-scale).
    gnn_preds = knn_transfer(X_gnn[train_idx], y_train, X_gnn[test_idx], k=k)
    gnn_acc = float(np.mean([p == t for p, t in zip(gnn_preds, y_test, strict=True)]))

    return {
        "esm2_transfer": _scaled_knn(X_esm),
        "esm_gnn_fusion": gnn_acc,
        "catalyst_microenvironment": _scaled_knn(X_full),
    }


def main() -> int:
    ensure_dirs()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    seeds = [int(x) for x in os.environ.get("V05_SEEDS", "7,11,13").split(",") if x.strip()]
    epochs = int(os.environ.get("V05_EPOCHS", "200"))
    skip_train = os.environ.get("V05_SKIP_TRAIN", "").strip() in {"1", "true", "yes"}
    resume = os.environ.get("V05_RESUME", "1").strip() not in {"0", "false", "no"}

    if not (PROCESSED / "embedding_esm.npy").exists():
        print("error: missing embedding_esm.npy — run cat-esm first", file=sys.stderr)
        return 1

    from catalyst_atlas.models.train_encoder import train_reaction_center_encoder

    rows: list[dict] = []
    for seed in seeds:
        seed_emb = PROCESSED / f"embedding_esm_gnn_seed{seed}.npy"
        seed_ckpt = ARTIFACTS / f"reaction_center_esm_gnn_seed{seed}.pt"
        _log(f"=== seed {seed} ===")

        if not skip_train:
            if resume and seed_emb.exists():
                _log(f"  resume: using existing {seed_emb.name}")
                shutil.copy2(seed_emb, PROCESSED / "embedding_esm_gnn.npy")
                if seed_ckpt.exists():
                    shutil.copy2(seed_ckpt, ARTIFACTS / "reaction_center_esm_gnn.pt")
            else:
                _log(f"  train ESM+GNN epochs={epochs}")
                train_reaction_center_encoder(
                    split="fold_cluster",
                    epochs=epochs,
                    batch_size=32,
                    lr=3e-3,
                    seed=seed,
                    n_val_folds=12,
                    lambda_cls=0.3,
                    fusion_esm=True,
                    no_early_stop=True,
                    checkpoint_every=10,
                )
                for src, dst in [
                    (PROCESSED / "embedding_esm_gnn.npy", seed_emb),
                    (ARTIFACTS / "reaction_center_esm_gnn.pt", seed_ckpt),
                ]:
                    if src.exists():
                        shutil.copy2(src, dst)
                        _log(f"  wrote {dst.name}")
        elif seed_emb.exists():
            shutil.copy2(seed_emb, PROCESSED / "embedding_esm_gnn.npy")

        _log("  fold_cluster eval")
        scores = _fold_cluster_scores(seed)
        row = {"seed": seed, **scores}
        rows.append(row)
        _log(
            f"  fold_cluster  esm2={row['esm2_transfer']:.4f}  "
            f"esm_gnn={row['esm_gnn_fusion']:.4f}  eng={row['catalyst_microenvironment']:.4f}"
        )

        # Checkpoint summary after each seed so a crash still leaves numbers.
        summary = {
            "protocol": (
                "For each seed: train ESM+GNN with that seed, fold_cluster-only eval "
                "with the same seed. ESM2 uses fixed embedding_esm.npy (split variance only)."
            ),
            "seeds": seeds,
            "epochs": epochs,
            "skip_train": skip_train,
            "resume": resume,
            "per_seed": rows,
            "fold_cluster": {
                "esm2_transfer": _mean_std(
                    [r["esm2_transfer"] for r in rows if r["esm2_transfer"] is not None]
                ),
                "esm_gnn_fusion": _mean_std(
                    [r["esm_gnn_fusion"] for r in rows if r["esm_gnn_fusion"] is not None]
                ),
                "catalyst_microenvironment": _mean_std(
                    [
                        r["catalyst_microenvironment"]
                        for r in rows
                        if r["catalyst_microenvironment"] is not None
                    ]
                ),
            },
        }
        out = REPORTS / "v05_seed_summary.json"
        out.write_text(json.dumps(summary, indent=2))
        _log(f"  checkpoint summary → {out}")

    _log(f"wrote {REPORTS / 'v05_seed_summary.json'}")
    for name, block in summary["fold_cluster"].items():
        _log(f"  {name}: {block['mean']:.3f} ± {block['std']:.3f} (n={block['n']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
