#!/usr/bin/env python3
"""Multi-seed fold_cluster bake-off for ESM-2 vs ESM+GNN.

Protocol (seeds 7, 11, 13):
  train ESM+GNN --fusion-esm --seed S
  cat-eval --seed S --no-external
  record fold_cluster for esm2_transfer and esm_gnn_fusion

ESM2 variance is split sensitivity (embedding fixed).
ESM+GNN variance couples train init + split.

Env:
  V05_SKIP_TRAIN=1  — eval only with existing embedding_esm_gnn.npy (smoke)
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


def _acc(results: dict, method: str) -> float | None:
    m = results["splits"]["fold_cluster"]["methods"].get(method)
    return None if m is None else float(m["accuracy"])


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


def main() -> int:
    ensure_dirs()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    seeds = [int(x) for x in os.environ.get("V05_SEEDS", "7,11,13").split(",") if x.strip()]
    epochs = int(os.environ.get("V05_EPOCHS", "200"))
    skip_train = os.environ.get("V05_SKIP_TRAIN", "").strip() in {"1", "true", "yes"}

    if not (PROCESSED / "embedding_esm.npy").exists():
        print("error: missing embedding_esm.npy — run cat-esm first", file=sys.stderr)
        return 1

    from catalyst_atlas.eval.run import run_eval
    from catalyst_atlas.models.train_encoder import train_reaction_center_encoder

    rows: list[dict] = []
    for seed in seeds:
        print(f"=== seed {seed} ===")
        if not skip_train:
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
            # Keep per-seed copies so later runs do not clobber.
            for src, dst in [
                (PROCESSED / "embedding_esm_gnn.npy", PROCESSED / f"embedding_esm_gnn_seed{seed}.npy"),
                (
                    ARTIFACTS / "reaction_center_esm_gnn.pt",
                    ARTIFACTS / f"reaction_center_esm_gnn_seed{seed}.pt",
                ),
            ]:
                if src.exists():
                    shutil.copy2(src, dst)

        results = run_eval(k=5, seed=seed, run_external=False)
        row = {
            "seed": seed,
            "esm2_transfer": _acc(results, "esm2_transfer"),
            "esm_gnn_fusion": _acc(results, "esm_gnn_fusion"),
            "catalyst_microenvironment": _acc(results, "catalyst_microenvironment"),
        }
        rows.append(row)
        print(
            f"  fold_cluster  esm2={row['esm2_transfer']}  "
            f"esm_gnn={row['esm_gnn_fusion']}  eng={row['catalyst_microenvironment']}"
        )

    summary = {
        "protocol": (
            "For each seed: train ESM+GNN with that seed, eval with the same seed. "
            "ESM2 uses fixed embedding_esm.npy (split variance only)."
        ),
        "seeds": seeds,
        "epochs": epochs,
        "skip_train": skip_train,
        "per_seed": rows,
        "fold_cluster": {
            "esm2_transfer": _mean_std([r["esm2_transfer"] for r in rows if r["esm2_transfer"] is not None]),
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
    print(f"wrote {out}")
    for name, block in summary["fold_cluster"].items():
        print(f"  {name}: {block['mean']:.3f} ± {block['std']:.3f} (n={block['n']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
