#!/usr/bin/env bash
# Train the reaction-center encoder on RunPod (or any CUDA box).
#
# Suggested pod: RTX 4090 / A40 community, PyTorch + CUDA template.
# Expected runtime: minutes–low hours on n≈959.
#
# Prerequisites on the pod:
#   - this repo checked out
#   - data/processed/features_full_meta.parquet (from cat-embed)
#   - data/processed/microenvironments.parquet (from cat-sites)
#
#   bash scripts/runpod_train.sh
#   EPOCHS=80 bash scripts/runpod_train.sh --split fold_cluster
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python -m pip install -U pip
# gpu extra pins numpy<2 for torch compatibility
python -m pip install -e ".[gpu]"

python - <<'PY'
import torch
print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
PY

if [[ ! -f data/processed/features_full_meta.parquet ]]; then
  echo "error: missing data/processed/features_full_meta.parquet — run cat-embed first" >&2
  exit 1
fi

if [[ ! -f data/processed/reaction_center_graphs.parquet ]]; then
  echo "Building reaction-center graphs..."
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
ls -lh data/processed/embedding_learned.npy artifacts/reaction_center_encoder.pt \
  artifacts/train_encoder_summary.json 2>/dev/null || true
echo
echo "Next on this box (or sync artifacts back and run locally):"
echo "  cat-eval --no-external   # or full cat-eval with MMseqs/Foldseek"
echo "Optional control:"
echo "  cat-esm && cat-eval"
