#!/usr/bin/env bash
# Train the reaction-center encoder on RunPod (or any CUDA box).
#
# Suggested pod: RTX 4090 / A40 community, PyTorch template.
# Expected runtime: minutes–hours on n≈959 (not days).
#
#   bash scripts/runpod_train.sh
#   bash scripts/runpod_train.sh --epochs 80 --split fold_cluster
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python -m pip install -U pip
python -m pip install -e ".[gpu]"

# Graphs are CPU; rebuild if missing.
if [[ ! -f data/processed/reaction_center_graphs.parquet ]]; then
  cat-graphs -v
fi

cat-train-encoder \
  --split fold_cluster \
  --epochs "${EPOCHS:-40}" \
  --batch-size "${BATCH:-32}" \
  --seed 7 \
  -v \
  "$@"

echo "Artifacts:"
ls -lh data/processed/embedding_learned.npy artifacts/reaction_center_encoder.pt 2>/dev/null || true
echo "Re-run cat-eval to score learned + (optional) ESM controls on hard holdouts."
