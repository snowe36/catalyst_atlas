#!/usr/bin/env python3
"""Train ESM+random-graph ablation and write fold_cluster comparison JSON.

Requires existing embedding_esm.npy and (for catalytic column) embedding_esm_gnn.npy.
Trains --fusion-esm --random-graphs with seed=7, then evals.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from catalyst_atlas.paths import PROCESSED, REPORTS, ensure_dirs  # noqa: E402


def main() -> int:
    ensure_dirs()
    REPORTS.mkdir(parents=True, exist_ok=True)
    epochs = int(os.environ.get("V05_EPOCHS", "200"))
    skip_train = os.environ.get("V05_SKIP_TRAIN", "").strip() in {"1", "true", "yes"}

    if not (PROCESSED / "embedding_esm.npy").exists():
        print("error: missing embedding_esm.npy", file=sys.stderr)
        return 1

    from catalyst_atlas.eval.run import run_eval
    from catalyst_atlas.models.train_encoder import train_reaction_center_encoder

    if not skip_train:
        print("=== train ESM + random graphs ===")
        train_reaction_center_encoder(
            split="fold_cluster",
            epochs=epochs,
            batch_size=32,
            lr=3e-3,
            seed=7,
            n_val_folds=12,
            lambda_cls=0.3,
            fusion_esm=True,
            random_graphs=True,
            no_early_stop=True,
            checkpoint_every=10,
        )

    results = run_eval(k=5, seed=7, run_external=False)
    fold = results["splits"]["fold_cluster"]["methods"]

    def acc(name: str) -> float | None:
        m = fold.get(name)
        return None if m is None else float(m["accuracy"])

    summary = {
        "seed": 7,
        "epochs": epochs,
        "fold_cluster": {
            "esm2_transfer": acc("esm2_transfer"),
            "esm_gnn_random_graph": acc("esm_gnn_random_graph"),
            "esm_gnn_fusion": acc("esm_gnn_fusion"),
            "catalyst_microenvironment": acc("catalyst_microenvironment"),
        },
        "note": (
            "Catalytic graph should beat random-graph fusion if the +0.03 over ESM "
            "comes from chemistry in the microenvironment, not just extra params."
        ),
    }
    out = REPORTS / "v05_ablation_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")
    for k, v in summary["fold_cluster"].items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
