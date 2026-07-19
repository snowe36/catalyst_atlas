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
#   EPOCHS=200 bash scripts/runpod_train.sh --split fold_cluster --fusion
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

# Parse --fusion / --max-first-shell from passthrough; rebuild lean graphs by default.
DO_FUSION=0
MAX_SHELL="${MAX_FIRST_SHELL:-4}"
PASS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --fusion) DO_FUSION=1; shift ;;
    --max-first-shell)
      MAX_SHELL="$2"
      shift 2
      ;;
    --max-first-shell=*)
      MAX_SHELL="${1#*=}"
      shift
      ;;
    *) PASS+=("$1"); shift ;;
  esac
done

echo "Building lean reaction-center graphs (max_first_shell=${MAX_SHELL})..."
cat-graphs --max-first-shell "${MAX_SHELL}" -v

EPOCHS="${EPOCHS:-200}"
BATCH="${BATCH:-32}"
COMMON=(
  --split fold_cluster
  --epochs "${EPOCHS}"
  --batch-size "${BATCH}"
  --seed 7
  --patience "${PATIENCE:-30}"
  --val-folds "${VAL_FOLDS:-4}"
  --lambda-cls "${LAMBDA_CLS:-0.3}"
  -v
)

echo "Training learned GNN encoder..."
if [[ ${#PASS[@]} -gt 0 ]]; then
  cat-train-encoder "${COMMON[@]}" "${PASS[@]}"
else
  cat-train-encoder "${COMMON[@]}"
fi

if [[ "${DO_FUSION}" -eq 1 ]]; then
  echo "Training fusion encoder (GNN + engineered side vector)..."
  if [[ ${#PASS[@]} -gt 0 ]]; then
    cat-train-encoder "${COMMON[@]}" --fusion "${PASS[@]}"
  else
    cat-train-encoder "${COMMON[@]}" --fusion
  fi
fi

echo "Artifacts:"
ls -lh \
  data/processed/embedding_learned.npy \
  data/processed/embedding_fusion.npy \
  artifacts/reaction_center_encoder.pt \
  artifacts/reaction_center_fusion.pt \
  artifacts/train_encoder_summary.json \
  artifacts/train_fusion_summary.json \
  2>/dev/null || true
echo
echo "Next on this box (or sync artifacts back and run locally):"
echo "  cat-eval --no-external   # or full cat-eval with MMseqs/Foldseek"
echo "Optional control:"
echo "  cat-esm && cat-eval"
