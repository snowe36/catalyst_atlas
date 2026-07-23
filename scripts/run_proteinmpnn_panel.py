#!/usr/bin/env python3
"""Run ProteinMPNN shell-only redesign for exported panel jobs.

Expects:
  data/processed/design/mpnn_jobs/{enzyme_id}/pocket.json + {pdb}.pdb
  ProteinMPNN cloned at MPNN_DIR (default /workspace/ProteinMPNN)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "data" / "processed" / "design" / "mpnn_jobs"
OUT_FASTA = ROOT / "data" / "processed" / "design" / "mpnn_designs.fasta"
MPNN_DIR = Path(os.environ.get("MPNN_DIR", "/workspace/ProteinMPNN"))


def _extract_chain_pdb(src: Path, chain: str, dest: Path) -> Path:
    """Write a single-chain PDB so ProteinMPNN length matches our fixed map."""
    lines_out: list[str] = []
    for line in src.read_text(errors="replace").splitlines():
        if line.startswith(("ATOM", "HETATM")):
            if len(line) > 21 and line[21] == chain:
                lines_out.append(line)
        elif line.startswith("END"):
            break
        elif line.startswith(("HEADER", "TITLE", "CRYST", "SCALE", "ORIGX", "REMARK")):
            lines_out.append(line)
    lines_out.append("END")
    dest.write_text("\n".join(lines_out) + "\n")
    return dest


def _run_one(job: Path, n_seq: int) -> list[dict[str, str]]:
    pocket = json.loads((job / "pocket.json").read_text())
    fixed = json.loads((job / "fixed_positions.json").read_text())
    eid = pocket["enzyme_id"]
    chain = str(fixed.get("chain") or pocket.get("design_chain") or "A")
    pdb_id = str(pocket.get("pdb_id") or "").lower()
    pdb = job / f"{pdb_id}.pdb"
    if not pdb.exists():
        print(f"skip {eid}: missing pdb", file=sys.stderr)
        return []

    runner = MPNN_DIR / "protein_mpnn_run.py"
    if not runner.exists():
        raise FileNotFoundError(f"Missing {runner}")

    out_dir = job / "mpnn_out"
    out_dir.mkdir(exist_ok=True)
    chain_pdb = job / f"{pdb_id}_{chain}.pdb"
    _extract_chain_pdb(pdb, chain, chain_pdb)
    # ProteinMPNN jsonl: one JSON object per line keyed by pdb stem.
    fp_path = job / "fixed_positions.jsonl"
    fp_path.write_text(json.dumps({chain_pdb.stem: fixed["fixed_positions"]}) + "\n")

    cmd = [
        sys.executable,
        str(runner),
        "--pdb_path",
        str(chain_pdb),
        "--pdb_path_chains",
        chain,
        "--out_folder",
        str(out_dir),
        "--num_seq_per_target",
        str(n_seq),
        "--sampling_temp",
        "0.1",
        "--batch_size",
        "1",
        "--fixed_positions_jsonl",
        str(fp_path),
    ]
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(MPNN_DIR))

    # Collect sequences from seqs/*.fa
    records: list[dict[str, str]] = []
    for fa in (out_dir / "seqs").glob("*.fa"):
        header = None
        chunks: list[str] = []
        for line in fa.read_text().splitlines():
            if line.startswith(">"):
                if header is not None and chunks:
                    # Skip native/first sequence if tagged as score-only.
                    seq = "".join(chunks).replace(" ", "").upper()
                    if "model_name" in header or "T=" in header or "sample" in header.lower():
                        did = f"{eid}_mpnn_{len(records):04d}"
                        records.append({"enzyme_id": eid, "design_id": did, "sequence": seq})
                header = line[1:]
                chunks = []
            else:
                chunks.append(line.strip())
        if header is not None and chunks:
            seq = "".join(chunks).replace(" ", "").upper()
            if len(records) < n_seq:
                did = f"{eid}_mpnn_{len(records):04d}"
                records.append({"enzyme_id": eid, "design_id": did, "sequence": seq})
    # Drop WT-identical first hit if present.
    wt = pocket["sequence"]
    records = [r for r in records if r["sequence"] != wt][:n_seq]
    print(f"{eid}: {len(records)} designs", flush=True)
    return records


def main() -> int:
    n_seq = int(os.environ.get("N_SEQ", "100"))
    if not MPNN_DIR.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/dauparas/ProteinMPNN.git", str(MPNN_DIR)],
            check=True,
        )
    all_recs: list[dict[str, str]] = []
    for job in sorted(JOBS.iterdir()):
        if not job.is_dir() or not (job / "pocket.json").exists():
            continue
        try:
            all_recs.extend(_run_one(job, n_seq))
        except subprocess.CalledProcessError as exc:
            print(f"failed {job.name}: {exc}", file=sys.stderr)
    OUT_FASTA.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for r in all_recs:
        lines.append(f">{r['enzyme_id']}|{r['design_id']}")
        seq = r["sequence"]
        for i in range(0, len(seq), 80):
            lines.append(seq[i : i + 80])
    OUT_FASTA.write_text("\n".join(lines) + ("\n" if lines else ""))
    print(f"wrote {len(all_recs)} sequences → {OUT_FASTA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
