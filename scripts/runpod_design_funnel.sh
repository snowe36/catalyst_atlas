#!/usr/bin/env bash
# Run ProteinMPNN shell redesign + ColabFold on the AF shortlist (RunPod GPU).
#
# Expected layout on the pod:
#   /workspace/catalyst_atlas   (this repo)
#   /workspace/ProteinMPNN      (https://github.com/dauparas/ProteinMPNN)
#
# Usage (after syncing panel pockets + PDBs):
#   bash scripts/runpod_design_funnel.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python -m pip install -U pip
python -m pip install -e ".[gpu]"
# ColabFold 1.6 + dm-haiku need JAX 0.4.x CUDA wheels (not jax 0.11 CPU/CPU-break).
python -m pip install -U "jax[cuda12]==0.4.38"

N_SEQ="${N_SEQ:-100}"
TOP_K="${TOP_K:-10}"
PANEL_JSON="${PANEL_JSON:-data/processed/design/panel.json}"

if [[ ! -f "$PANEL_JSON" ]]; then
  echo "error: missing $PANEL_JSON — build panel locally first" >&2
  exit 1
fi

python - <<'PY'
import json, torch
from pathlib import Path
print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
panel = json.loads(Path("data/processed/design/panel.json").read_text())
print("panel", [p["enzyme_id"] for p in panel])
PY

MPNN_DIR="${MPNN_DIR:-/workspace/ProteinMPNN}"
if [[ ! -d "$MPNN_DIR" ]]; then
  git clone --depth 1 https://github.com/dauparas/ProteinMPNN.git "$MPNN_DIR"
fi

echo "=== ProteinMPNN jobs under data/processed/design/mpnn_jobs ==="
python - <<PY
import json
from pathlib import Path
from catalyst_atlas.design.pocket import load_pocket
from catalyst_atlas.design.mpnn import export_mpnn_job

panel = json.loads(Path("$PANEL_JSON").read_text())
for p in panel:
    pocket = load_pocket(p["enzyme_id"])
    export_mpnn_job(pocket, n_sequences=int("$N_SEQ"))
    print("exported", p["enzyme_id"])
PY

# Minimal ProteinMPNN wrapper: design redesignable positions only by fixing catalytic.
# Full ProteinMPNN CLI varies by version; write a thin runner.
python - <<'PY'
"""Run ProteinMPNN per exported job if protein_mpnn_run.py exists."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

mpnn = Path("/workspace/ProteinMPNN")
runner = mpnn / "protein_mpnn_run.py"
jobs = Path("data/processed/design/mpnn_jobs")
if not runner.exists():
    print("ProteinMPNN runner missing; skip GPU generate — import FASTA later", file=sys.stderr)
    sys.exit(0)

out_fasta = Path("data/processed/design/mpnn_all.fasta")
lines = []
for job in sorted(jobs.iterdir()):
    if not job.is_dir():
        continue
    pocket = json.loads((job / "pocket.json").read_text())
    fixed = json.loads((job / "fixed_positions.json").read_text())
    pdb_id = (pocket.get("pdb_id") or "").lower()
    pdb = job / f"{pdb_id}.pdb"
    if not pdb.exists():
        print("skip (no pdb)", job.name)
        continue
    out_dir = job / "mpnn_out"
    out_dir.mkdir(exist_ok=True)
    # ProteinMPNN expects a folder of PDBs; copy single pdb.
    pdb_dir = job / "pdb_in"
    pdb_dir.mkdir(exist_ok=True)
    target = pdb_dir / pdb.name
    if not target.exists():
        target.write_bytes(pdb.read_bytes())
    # Fixed positions JSON in ProteinMPNN format: {pdbname: {chain: [resnums]}}
    fp = {pdb.stem: fixed["fixed_positions"]}
    fp_path = job / "fixed_pos_mpnn.json"
    fp_path.write_text(json.dumps(fp))
    cmd = [
        sys.executable, str(runner),
        "--pdb_path_chains", "A",
        "--out_folder", str(out_dir),
        "--num_seq_per_target", str(json.loads((job / "manifest.json").read_text())["n_sequences"]),
        "--sampling_temp", "0.1",
        "--batch_size", "1",
        "--pdb_path", str(target),
        "--fixed_positions_jsonl", str(fp_path),
    ]
    # Some ProteinMPNN versions use jsonl differently; try and continue on failure.
    print("running", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print("mpnn failed", job.name, exc, file=sys.stderr)
        continue
print("done mpnn loop")
PY

echo "=== Funnel shortlist (after designs imported) ==="
echo "On the workstation after syncing designs.fasta:"
echo "  cat-design-generate --from-sequences data/processed/design/designs.fasta"
echo "  cat-design-funnel --top-k ${TOP_K}"
echo "  # then ColabFold af_queue.fasta"
echo "Done prep on pod."
