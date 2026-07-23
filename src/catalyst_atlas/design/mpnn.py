"""ProteinMPNN adapter — export fixed-position jobs; import designed sequences.

Generation is optional/external. Evaluation never imports this module's runner.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

from catalyst_atlas.paths import PROCESSED, ensure_dirs

logger = logging.getLogger(__name__)


def fixed_positions_from_pocket(pocket: dict[str, Any]) -> dict[str, Any]:
    """Build ProteinMPNN fixed-position maps for shell-only redesign.

    ProteinMPNN designs every non-fixed residue. For Option B we therefore
    **fix all residues except the redesignable shell** (catalytic included).

    Numbers are **PDB residue numbers** per chain.
    """
    chain = str(pocket.get("design_chain") or "A")
    redesignable = {
        int(r["resnum"])
        for r in (pocket.get("redesignable") or [])
        if str(r.get("chain") or "A") == chain
    }
    # All CA positions present in the design sequence mapping.
    all_resnums = {
        int(r["resnum"])
        for r in (pocket.get("catalytic_residues") or [])
        if str(r.get("chain") or "A") == chain
    } | redesignable
    # Prefer explicit chain length from seq_index map if present on residues.
    for r in list(pocket.get("catalytic_residues") or []) + list(
        pocket.get("redesignable") or []
    ):
        if str(r.get("chain") or "A") == chain:
            all_resnums.add(int(r["resnum"]))

    # Expand to full chain resnums when pocket stored them via redesignable+catalytic
    # only — MPNN still needs every non-shell residue fixed. Reconstruct from
    # sequence length + seq_index↔resnum pairs when available.
    idx_to_resnum: dict[int, int] = {}
    for r in list(pocket.get("catalytic_residues") or []) + list(
        pocket.get("redesignable") or []
    ):
        if r.get("seq_index") is None:
            continue
        if str(r.get("chain") or "A") != chain:
            continue
        idx_to_resnum[int(r["seq_index"])] = int(r["resnum"])

    # If we only know shell/catalytic resnums, fix those we know must stay and
    # also emit designed_positions for the shell (helper for runners).
    known = sorted(all_resnums)
    fixed = [n for n in known if n not in redesignable]

    # When sequence is PDB-derived, recover full chain resnum list by walking
    # contiguous indices if we can invert a dense map; else fix catalytic +
    # everything except redesignable among known positions and rely on the
    # runner to fix the full chain from the PDB.
    chain_resnums = pocket.get("chain_resnums")
    if chain_resnums:
        full = [int(n) for n in chain_resnums]
        fixed = [n for n in full if n not in redesignable]

    return {
        "enzyme_id": pocket["enzyme_id"],
        "chain": chain,
        "fixed_positions": {chain: sorted(set(fixed))},
        "designed_positions": {chain: sorted(redesignable)},
        "redesignable_seq_indices_0based": sorted(
            {
                int(r["seq_index"])
                for r in (pocket.get("redesignable") or [])
                if r.get("seq_index") is not None
            }
        ),
        "note": "Fix everything except redesignable shell (Option B).",
    }


def export_mpnn_job(
    pocket: dict[str, Any],
    *,
    out_dir: Path | None = None,
    n_sequences: int = 100,
) -> Path:
    """Write a job directory: pocket copy, fixed positions, run manifest."""
    ensure_dirs()
    eid = pocket["enzyme_id"]
    job_dir = out_dir or (PROCESSED / "design" / "mpnn_jobs" / eid)
    job_dir.mkdir(parents=True, exist_ok=True)

    (job_dir / "pocket.json").write_text(json.dumps(pocket, indent=2))
    fixed = fixed_positions_from_pocket(pocket)
    (job_dir / "fixed_positions.json").write_text(json.dumps(fixed, indent=2))

    # Optional: copy cached PDB if present for external ProteinMPNN.
    pdb_id = str(pocket.get("pdb_id") or "").lower()
    pdb_src = PROCESSED.parent / "raw" / "pdb" / f"{pdb_id}.pdb"
    if pdb_id and pdb_src.exists():
        shutil.copy2(pdb_src, job_dir / f"{pdb_id}.pdb")

    manifest = {
        "enzyme_id": eid,
        "n_sequences": int(n_sequences),
        "sequence_length": len(pocket.get("sequence") or ""),
        "generator": "proteinmpnn",
        "note": (
            "Run ProteinMPNN externally with fixed_positions.json; "
            "write designs.fasta then cat-design-generate --from-sequences."
        ),
    }
    (job_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("Exported MPNN job for %s → %s", eid, job_dir)
    return job_dir


def run_proteinmpnn(
    job_dir: Path,
    *,
    proteinmpnn_script: str | Path | None = None,
    extra_args: list[str] | None = None,
) -> Path:
    """Invoke an external ProteinMPNN script if configured.

    Raises FileNotFoundError when no runner is available — use FASTA import instead.
    """
    script = proteinmpnn_script
    if script is None:
        raise FileNotFoundError(
            "No ProteinMPNN script configured. Export the job with export_mpnn_job, "
            "run ProteinMPNN externally, then import via parse_design_fasta."
        )
    script = Path(script)
    if not script.exists():
        raise FileNotFoundError(f"ProteinMPNN script not found: {script}")

    cmd = ["python", str(script), "--job-dir", str(job_dir)]
    if extra_args:
        cmd.extend(extra_args)
    logger.info("Running ProteinMPNN: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    fasta = job_dir / "designs.fasta"
    if not fasta.exists():
        raise FileNotFoundError(f"Expected designs.fasta under {job_dir}")
    return fasta


def parse_design_fasta(path: Path) -> list[dict[str, str]]:
    """Parse a simple FASTA of designed sequences.

    Headers: ``>enzyme_id|design_id`` or ``>design_id``.
    """
    records: list[dict[str, str]] = []
    design_id: str | None = None
    enzyme_id = ""
    chunks: list[str] = []

    def _flush() -> None:
        nonlocal design_id, enzyme_id, chunks
        if design_id is None:
            return
        records.append(
            {
                "enzyme_id": enzyme_id,
                "design_id": design_id,
                "sequence": "".join(chunks).replace(" ", "").upper(),
            }
        )
        design_id = None
        enzyme_id = ""
        chunks = []

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            _flush()
            header = line[1:].strip()
            if "|" in header:
                enzyme_id, design_id = header.split("|", 1)
            else:
                design_id = header
                enzyme_id = ""
            chunks = []
        else:
            chunks.append(line)
    _flush()
    return records


def write_design_fasta(records: list[dict[str, str]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for rec in records:
        eid = rec.get("enzyme_id") or "unknown"
        did = rec.get("design_id") or "design"
        lines.append(f">{eid}|{did}")
        seq = rec["sequence"]
        for i in range(0, len(seq), 80):
            lines.append(seq[i : i + 80])
    path.write_text("\n".join(lines) + "\n")
    return path
