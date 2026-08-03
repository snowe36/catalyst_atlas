#!/usr/bin/env bash
# 5-minute showcase — synthetic atlas + mock design (no M-CSA / ColabFold)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON="$ROOT/.venv/bin/python"
  elif command -v python3.11 >/dev/null 2>&1; then
    PYTHON=python3.11
  else
    PYTHON=python3
  fi
else
  PYTHON=python
fi

export MPLBACKEND=Agg
"$PYTHON" -m pip install -q -e ".[dev]"
"$PYTHON" -c "from catalyst_atlas.demo import run_demo; run_demo()"

echo ""
echo "Open:"
echo "  demo/outputs/fig1_pipeline.png"
echo "  demo/outputs/fig_design_landscape.png"
echo "  demo/outputs/design_case_study.md"
