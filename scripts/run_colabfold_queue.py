#!/usr/bin/env python3
"""Run ColabFold on af_queue.fasta and write design.predict metrics JSON.

Funnel-friendly MSA strategy (default ``reuse_wt``):
  1. Predict each WT with ColabFold MSA (once per enzyme).
  2. Rebuild design A3Ms by swapping the WT query sequence for the design.
  3. Predict designs from those A3Ms (no MSA server per mutant).

Usage on GPU box:
  pip install 'colabfold[alphafold-minus-jax]@git+https://github.com/sokrypton/ColabFold'
  # pin CUDA JAX for ColabFold 1.6 / dm-haiku:
  pip install 'jax[cuda12]==0.4.38'
  python scripts/run_colabfold_queue.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FASTA = ROOT / "data" / "processed" / "design" / "af_queue.fasta"
OUT = ROOT / "data" / "processed" / "design" / "colabfold_out"
PRED = ROOT / "data" / "processed" / "design" / "predictions"
A3M_DIR = ROOT / "data" / "processed" / "design" / "colabfold_a3m"


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


def _colabfold_bin() -> str:
    colabfold = shutil.which("colabfold_batch") or "/workspace/venv/bin/colabfold_batch"
    if not Path(colabfold).exists():
        raise FileNotFoundError(
            "colabfold_batch not found. Install LocalColabFold or colabfold CLI."
        )
    return colabfold


def _run_colabfold(inputs: Path, out: Path, *, msa_mode: str | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        _colabfold_bin(),
        str(inputs),
        str(out),
        "--num-models",
        "1",
        "--num-recycle",
        "1",
        "--model-type",
        "alphafold2_ptm",
    ]
    if msa_mode:
        cmd.extend(["--msa-mode", msa_mode])
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _write_fasta(recs: list[tuple[str, str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for eid, did, seq in recs:
        lines.append(f">{eid}|{did}")
        for i in range(0, len(seq), 80):
            lines.append(seq[i : i + 80])
    path.write_text("\n".join(lines) + "\n")


def _find_wt_a3m(eid: str) -> Path | None:
    # ColabFold names: MCSA00032_WT.a3m
    candidates = [
        OUT / f"{eid}_WT.a3m",
        OUT / f"{eid}|WT.a3m",
    ]
    for c in candidates:
        if c.exists() and c.stat().st_size > 0:
            return c
    hits = list(OUT.glob(f"{eid}*WT*.a3m"))
    return hits[0] if hits else None


def _a3m_with_query(wt_a3m: Path, query_name: str, query_seq: str) -> str:
    """Replace the first (query) sequence in an A3M with query_seq; keep hits.

    ColabFold A3Ms often start with a ``#length\\tnseq`` comment line before
    the first ``>`` header — skip those.
    """
    lines = wt_a3m.read_text(errors="replace").splitlines()
    i = 0
    while i < len(lines) and not lines[i].startswith(">"):
        i += 1
    if i >= len(lines):
        raise ValueError(f"bad a3m (no header): {wt_a3m}")
    # skip first record (query + its sequence lines)
    j = i + 1
    while j < len(lines) and not lines[j].startswith(">"):
        j += 1
    rest = lines[j:]
    out = [f">{query_name}"]
    for k in range(0, len(query_seq), 80):
        out.append(query_seq[k : k + 80])
    out.extend(rest)
    return "\n".join(out) + "\n"


def _write_metrics_from_pdbs() -> int:
    n = 0
    for pdb in OUT.rglob("*.pdb"):
        name = pdb.stem
        m = re.match(r"(.+?)[\|_](.+?)(?:_unrelaxed|_relaxed|_rank).*$", name)
        if not m:
            parts = name.split("_")
            if len(parts) < 2:
                continue
            eid = parts[0]
            did = "_".join(parts[1:]).split("_unrelaxed")[0].split("_rank")[0]
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
    return n


def _strategy_reuse_wt(recs: list[tuple[str, str, str]]) -> None:
    wts = [(e, d, s) for e, d, s in recs if d == "WT"]
    designs = [(e, d, s) for e, d, s in recs if d != "WT"]
    print(f"reuse_wt: {len(wts)} WT + {len(designs)} designs", flush=True)

    # 1) WT MSAs + structures (skip already-done)
    pending_wt = []
    for eid, did, seq in wts:
        done = OUT / f"{eid}_WT.done.txt"
        if done.exists() and _find_wt_a3m(eid) is not None:
            print(f"skip WT {eid} (done)", flush=True)
            continue
        pending_wt.append((eid, did, seq))
    if pending_wt:
        wt_fa = OUT / "_wt_only.fasta"
        _write_fasta(pending_wt, wt_fa)
        _run_colabfold(wt_fa, OUT, msa_mode="mmseqs2_uniref_env")

    # 2) Design A3Ms from WT MSA
    A3M_DIR.mkdir(parents=True, exist_ok=True)
    n_a3m = 0
    n_skip = 0
    missing_wt_msa: set[str] = set()
    for eid, did, seq in designs:
        # Skip if this design already has a PDB
        existing = list(OUT.glob(f"{eid}_{did}*.pdb")) + list(
            OUT.glob(f"{eid}|{did}*.pdb")
        )
        if existing:
            n_skip += 1
            continue
        wt_a3m = _find_wt_a3m(eid)
        if wt_a3m is None:
            missing_wt_msa.add(eid)
            continue
        stem = f"{eid}_{did}"
        dest = A3M_DIR / f"{stem}.a3m"
        dest.write_text(_a3m_with_query(wt_a3m, stem, seq))
        n_a3m += 1
    if missing_wt_msa:
        print(f"WARNING: no WT MSA for {sorted(missing_wt_msa)}", flush=True)
    print(
        f"wrote {n_a3m} design a3ms → {A3M_DIR} (skipped {n_skip} already folded)",
        flush=True,
    )

    # 3) Predict designs from A3M dir (ColabFold treats a3m as MSA input)
    if n_a3m:
        _run_colabfold(A3M_DIR, OUT, msa_mode=None)
    elif n_skip:
        print("all designs already folded; skip colabfold_batch", flush=True)


def main() -> int:
    if not FASTA.exists():
        print(f"missing {FASTA}", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    PRED.mkdir(parents=True, exist_ok=True)

    strategy = os.environ.get("COLABFOLD_STRATEGY", "reuse_wt")
    recs = _parse_fasta(FASTA)
    if not recs:
        print("empty fasta", file=sys.stderr)
        return 1

    try:
        if strategy == "reuse_wt":
            _strategy_reuse_wt(recs)
        else:
            msa_mode = os.environ.get("COLABFOLD_MSA_MODE", "mmseqs2_uniref_env")
            _run_colabfold(FASTA, OUT, msa_mode=msa_mode)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    n = _write_metrics_from_pdbs()
    print(f"wrote metrics for {n} models → {PRED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
