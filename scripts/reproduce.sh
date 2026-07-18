#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python -m pip install -e ".[dev]" >/dev/null
cat-download --demo --n-enzymes 800
cat-sites
cat-embed
cat-eval
cat-search --demo-hero
cat-figures
pytest -q
