#!/usr/bin/env python3
"""Run remaining v05 seed trains (+ optional ablation) in parallel on one GPU.

Bottleneck is Python graph-encode overhead (~12% GPU util, ~0.5GB VRAM), so a
faster GPU barely helps — parallel seed jobs on the same 4090 do.

Env:
  V05_SEEDS=11,13
  V05_EPOCHS=200
  V05_PARALLEL_ABLATION=1  — also train random-graph ablation
  V05_SKIP_EXISTING=1      — skip seeds that already have embeddings (default)
"""

from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from catalyst_atlas.paths import PROCESSED, REPORTS, ensure_dirs  # noqa: E402

ARTIFACTS = ROOT / "artifacts"


def env_path_rest() -> str:
    """Keep existing PYTHONPATH entries except a shadowing repo-root package dir."""
    parts = []
    for p in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if not p:
            continue
        if Path(p).resolve() == ROOT.resolve():
            continue
        parts.append(p)
    return os.pathsep.join(parts)


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


def _train_seed(seed: int, epochs: int) -> dict[str, str | int]:
    tag = f"seed{seed}"
    emb = PROCESSED / f"embedding_esm_gnn_{tag}.npy"
    log_path = ROOT / f"v05_train_{tag}.log"
    _log(f"[seed {seed}] start → {log_path.name}")
    cmd = [
        sys.executable,
        "-c",
        (
            "from catalyst_atlas.models.train_encoder import train_reaction_center_encoder\n"
            f"train_reaction_center_encoder("
            f"split='fold_cluster', epochs={epochs}, batch_size=32, lr=3e-3, "
            f"seed={seed}, n_val_folds=12, lambda_cls=0.3, fusion_esm=True, "
            f"no_early_stop=True, checkpoint_every=10, run_tag='{tag}')\n"
            "print('TRAIN_OK', flush=True)\n"
        ),
    ]
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        # Prefer src/ layout; a flat catalyst_atlas/ package on the pod shadows it.
        "PYTHONPATH": str(ROOT / "src") + os.pathsep + env_path_rest(),
    }
    with log_path.open("w") as logf:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0 or not emb.exists():
        raise RuntimeError(f"seed {seed} train failed rc={proc.returncode} emb={emb.exists()}")
    # Also keep untagged copy for the last finisher (harmless).
    shutil.copy2(emb, PROCESSED / "embedding_esm_gnn.npy")
    ckpt = ARTIFACTS / f"reaction_center_esm_gnn_{tag}.pt"
    if ckpt.exists():
        shutil.copy2(ckpt, ARTIFACTS / "reaction_center_esm_gnn.pt")
    _log(f"[seed {seed}] done")
    return {"seed": seed, "embedding": str(emb), "log": str(log_path)}


def _train_ablation(epochs: int) -> dict[str, str]:
    emb = PROCESSED / "embedding_esm_gnn_randnodes.npy"
    log_path = ROOT / "v05_train_ablation.log"
    _log(f"[ablation] start → {log_path.name}")
    cmd = [
        sys.executable,
        "-c",
        (
            "from catalyst_atlas.models.train_encoder import train_reaction_center_encoder\n"
            f"train_reaction_center_encoder("
            f"split='fold_cluster', epochs={epochs}, batch_size=32, lr=3e-3, "
            f"seed=7, n_val_folds=12, lambda_cls=0.3, fusion_esm=True, "
            f"random_graphs=True, no_early_stop=True, checkpoint_every=10)\n"
            "print('TRAIN_OK', flush=True)\n"
        ),
    ]
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(ROOT / "src") + os.pathsep + env_path_rest(),
    }
    with log_path.open("w") as logf:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0 or not emb.exists():
        raise RuntimeError(f"ablation train failed rc={proc.returncode} emb={emb.exists()}")
    _log("[ablation] done")
    return {"embedding": str(emb), "log": str(log_path)}


def main() -> int:
    ensure_dirs()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    seeds = [int(x) for x in os.environ.get("V05_SEEDS", "11,13").split(",") if x.strip()]
    epochs = int(os.environ.get("V05_EPOCHS", "200"))
    do_ablation = os.environ.get("V05_PARALLEL_ABLATION", "1").strip() not in {
        "0",
        "false",
        "no",
    }
    skip_existing = os.environ.get("V05_SKIP_EXISTING", "1").strip() not in {
        "0",
        "false",
        "no",
    }

    if not (PROCESSED / "embedding_esm.npy").exists():
        print("error: missing embedding_esm.npy", file=sys.stderr)
        return 1

    # Import fold helper from sequential bakeoff script.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "v05_seed_bakeoff", ROOT / "scripts" / "v05_seed_bakeoff.py"
    )
    bake = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(bake)

    jobs: list[tuple[str, callable]] = []
    for seed in seeds:
        emb = PROCESSED / f"embedding_esm_gnn_seed{seed}.npy"
        if skip_existing and emb.exists():
            _log(f"[seed {seed}] skip (exists)")
            continue
        jobs.append((f"seed{seed}", lambda s=seed: _train_seed(s, epochs)))

    if do_ablation:
        ab_emb = PROCESSED / "embedding_esm_gnn_randnodes.npy"
        if skip_existing and ab_emb.exists():
            _log("[ablation] skip (exists)")
        else:
            jobs.append(("ablation", lambda: _train_ablation(epochs)))

    _log(f"=== V05 PARALLEL START seeds={seeds} jobs={len(jobs)} epochs={epochs} ===")
    t0 = time.time()
    errors: list[str] = []
    if jobs:
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            futs = {pool.submit(fn): name for name, fn in jobs}
            for fut in as_completed(futs):
                name = futs[fut]
                try:
                    fut.result()
                except Exception as exc:  # noqa: BLE001 — surface worker failures
                    errors.append(f"{name}: {exc}")
                    _log(f"ERROR {name}: {exc}")

    if errors:
        _log(f"parallel trains failed: {errors}")
        return 1

    _log(f"=== trains finished in {time.time() - t0:.0f}s — fold_cluster evals ===")

    # Eval every seed that has an embedding (include seed 7 if present).
    eval_seeds = sorted(
        {
            int(p.name.replace("embedding_esm_gnn_seed", "").replace(".npy", ""))
            for p in PROCESSED.glob("embedding_esm_gnn_seed*.npy")
        }
        | set(seeds)
    )
    rows: list[dict] = []
    for seed in eval_seeds:
        emb = PROCESSED / f"embedding_esm_gnn_seed{seed}.npy"
        if not emb.exists():
            _log(f"[eval seed {seed}] missing embedding, skip")
            continue
        shutil.copy2(emb, PROCESSED / "embedding_esm_gnn.npy")
        meta_src = PROCESSED / f"embedding_esm_gnn_seed{seed}_meta.parquet"
        if meta_src.exists():
            shutil.copy2(meta_src, PROCESSED / "embedding_esm_gnn_meta.parquet")
        scores = bake._fold_cluster_scores(seed)
        row = {"seed": seed, **scores}
        rows.append(row)
        _log(
            f"[eval seed {seed}] esm2={row['esm2_transfer']:.4f} "
            f"esm_gnn={row['esm_gnn_fusion']:.4f} eng={row['catalyst_microenvironment']:.4f}"
        )

    summary = {
        "protocol": "Parallel ESM+GNN trains per seed; fold_cluster-only eval after.",
        "seeds": [r["seed"] for r in rows],
        "epochs": epochs,
        "parallel": True,
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
    _log(f"wrote {out}")
    for name, block in summary["fold_cluster"].items():
        _log(f"  {name}: {block['mean']:.3f} ± {block['std']:.3f} (n={block['n']})")

    # Ablation summary (uses catalytic seed7 if present, else first available).
    if (PROCESSED / "embedding_esm_gnn_randnodes.npy").exists():
        import numpy as np
        from sklearn.preprocessing import StandardScaler

        from catalyst_atlas.eval.baselines import knn_transfer
        from catalyst_atlas.eval.labels import chemistry_label_col
        from catalyst_atlas.eval.run import _align_embedding, _load_unscaled_features
        from catalyst_atlas.eval.splits import make_splits

        cat_seed = PROCESSED / "embedding_esm_gnn_seed7.npy"
        if cat_seed.exists():
            shutil.copy2(cat_seed, PROCESSED / "embedding_esm_gnn.npy")
            meta7 = PROCESSED / "embedding_esm_gnn_seed7_meta.parquet"
            if meta7.exists():
                shutil.copy2(meta7, PROCESSED / "embedding_esm_gnn_meta.parquet")

        meta, X_full, _ = _load_unscaled_features()
        label_col = chemistry_label_col(meta)
        X_esm = _align_embedding(
            meta, PROCESSED / "embedding_esm.npy", PROCESSED / "embedding_esm_meta.parquet"
        )
        X_gnn = _align_embedding(
            meta,
            PROCESSED / "embedding_esm_gnn.npy",
            PROCESSED / "embedding_esm_gnn_meta.parquet",
        )
        X_rand = _align_embedding(
            meta,
            PROCESSED / "embedding_esm_gnn_randnodes.npy",
            PROCESSED / "embedding_esm_gnn_randnodes_meta.parquet",
        )
        train_idx, test_idx = make_splits(meta, test_size=0.2, seed=7, label_col=label_col)[
            "fold_cluster"
        ]
        y_train = meta.iloc[train_idx][label_col].astype(str).tolist()
        y_test = meta.iloc[test_idx][label_col].astype(str).tolist()

        def scaled(X):
            sc = StandardScaler()
            tr, te = sc.fit_transform(X[train_idx]), sc.transform(X[test_idx])
            preds = knn_transfer(tr, y_train, te, k=5)
            return float(np.mean([p == t for p, t in zip(preds, y_test, strict=True)]))

        def unscaled(X):
            preds = knn_transfer(X[train_idx], y_train, X[test_idx], k=5)
            return float(np.mean([p == t for p, t in zip(preds, y_test, strict=True)]))

        ab = {
            "seed": 7,
            "epochs": epochs,
            "fold_cluster": {
                "esm2_transfer": scaled(X_esm) if X_esm is not None else None,
                "esm_gnn_random_graph": unscaled(X_rand) if X_rand is not None else None,
                "esm_gnn_fusion": unscaled(X_gnn) if X_gnn is not None else None,
                "catalyst_microenvironment": scaled(X_full),
            },
            "note": "Parallel ablation train; catalytic column from seed7 embedding when present.",
        }
        ab_path = REPORTS / "v05_ablation_summary.json"
        ab_path.write_text(json.dumps(ab, indent=2))
        _log(f"wrote {ab_path}")
        for k, v in ab["fold_cluster"].items():
            _log(f"  {k}: {v}")

    _log("=== V05 VALIDATION DONE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
