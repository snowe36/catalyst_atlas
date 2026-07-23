#!/usr/bin/env python3
"""Run ColabFold on af_queue.fasta and write design.predict metrics JSON.

Usage on GPU box:
  pip install 'colabfold[alphafold-minus-jax]@git+https://github.com/sokrypton/ColabFold'
  # or localcolabfold
  python scripts/run_colabfold_queue.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FASTA = ROOT / "data" / "processed" / "design" / "af_queue.fasta"
OUT = ROOT / "data" / "processed" / "design" / "colabfold_out"
PRED = ROOT / "data" / "processed" / "design" / "predictions"


def _parse_fasta(path: Path) -> list[tuple[str, str, str]]:
    recs: list[tuple[str, str, str]] = []
    eid = did = None
    chunks: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if eid and did and chunks:
                recs.append((eid, did, "".join(chunks)))
            header = line[1:].strip()
            if "|" in header:
                eid, did = header.split("|", 1)
            else:
                eid, did = "unknown", header
            chunks = []
        else:
            chunks.append(line.strip())
    if eid and did and chunks:
        recs.append((eid, did, "".join(chunks)))
    return recs


def _mean_plddt_from_pdb(pdb_path: Path) -> float | None:
    vals: list[float] = []
    for line in pdb_path.read_text(errors="replace").splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            try:
                vals.append(float(line[60:66]))
            except ValueError:
                continue
    if not vals:
        return None
    return sum(vals) / len(vals)


def main() -> int:
    if not FASTA.exists():
        print(f"missing {FASTA}", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    PRED.mkdir(parents=True, exist_ok=True)

    # Prefer colabfold_batch if available.
    cmd = [
        "colabfold_batch",
        str(FASTA),
        str(OUT),
        "--num-models",
        "1",
        "--num-recycle",
        "1",
        "--model-type",
        "alphafold2_ptm",
    ]
    print("RUN", " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print(
            "colabfold_batch not found. Install LocalColabFold or colabfold CLI.",
            file=sys.stderr,
        )
        return 2

    # Map outputs → metrics.json
    n = 0
    for pdb in OUT.rglob("*.pdb"):
        name = pdb.stem
        # Expect enzyme_id|design_id or enzyme_id_design_id in filename
        m = re.match(r"(.+?)[\|_](.+?)(?:_unrelaxed|_relaxed|_rank).*$", name)
        if not m:
            # ColabFold often uses header with underscores
            parts = name.split("_")
            if len(parts) < 2:
                continue
            eid, did = parts[0], "_".join(parts[1:]).split("_unrelaxed")[0].split("_rank")[0]
        else:
            eid, did = m.group(1), m.group(2)
        plddt = _mean_plddt_from_pdb(pdb)
        if plddt is None:
            continue
        dest = PRED / eid / did
        dest.mkdir(parents=True, exist_ok=True)
        metrics = {
            "enzyme_id": eid,
            "design_id": did,
            "mean_plddt": float(plddt),
            "pocket_pae": None,
            "pdb_path": str(pdb),
            "source": "colabfold",
        }
        (dest / "metrics.json").write_text(json.dumps(metrics, indent=2))
        n += 1
    print(f"wrote metrics for {n} models → {PRED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
