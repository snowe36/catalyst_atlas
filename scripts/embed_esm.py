#!/usr/bin/env python3
"""Frozen ESM-2 control embeddings (local or RunPod).

Usage:
  pip install -e '.[gpu]'
  python scripts/embed_esm.py
  python scripts/embed_esm.py --model esm2_t30_150M_UR50D --batch-size 2
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from catalyst_atlas.cli import esm_main

if __name__ == "__main__":
    raise SystemExit(esm_main())
