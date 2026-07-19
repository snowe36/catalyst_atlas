#!/usr/bin/env bash
# Train the reaction-center encoder on RunPod (or any CUDA box).
#
#   EPOCHS=250 bash scripts/runpod_train.sh --fusion-side
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python -m pip install -U pip
python -m pip install -e ".[gpu]"

python - <<'PY'
import torch
print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
PY

if [[ ! -f data/processed/features_full_meta.parquet ]]; then
  echo "error: missing data/processed/features_full_meta.parquet — run cat-embed first" >&2
  exit 1
fi

DO_FUSION=0
DO_FUSION_SIDE=0
MAX_SHELL="${MAX_FIRST_SHELL:-4}"
PASS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --fusion) DO_FUSION=1; shift ;;
    --fusion-side) DO_FUSION_SIDE=1; shift ;;
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

EPOCHS="${EPOCHS:-250}"
BATCH="${BATCH:-32}"
LR="${LR:-3e-3}"
COMMON=(
  --split fold_cluster
  --epochs "${EPOCHS}"
  --batch-size "${BATCH}"
  --lr "${LR}"
  --seed 7
  --patience "${PATIENCE:-50}"
  --val-folds "${VAL_FOLDS:-12}"
  --min-epochs "${MIN_EPOCHS:-100}"
  --lambda-cls "${LAMBDA_CLS:-0.3}"
  -v
)

echo "Training learned GNN encoder..."
if [[ ${#PASS[@]} -gt 0 ]]; then
  cat-train-encoder "${COMMON[@]}" "${PASS[@]}"
else
  cat-train-encoder "${COMMON[@]}"
fi

if [[ "${DO_FUSION_SIDE}" -eq 1 ]]; then
  echo "Training side-fusion encoder (GNN + metal/cofactor side vector)..."
  if [[ ${#PASS[@]} -gt 0 ]]; then
    cat-train-encoder "${COMMON[@]}" --fusion-side "${PASS[@]}"
  else
    cat-train-encoder "${COMMON[@]}" --fusion-side
  fi
elif [[ "${DO_FUSION}" -eq 1 ]]; then
  echo "Training fusion encoder (GNN + features_full)..."
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
echo "Next: cat-eval --no-external"
