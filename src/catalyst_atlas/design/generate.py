"""Generator-agnostic design orchestration (pocket → candidate sequences)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from catalyst_atlas.design.mpnn import (
    export_mpnn_job,
    parse_design_fasta,
    write_design_fasta,
)
from catalyst_atlas.design.pocket import load_pocket, run_pockets
from catalyst_atlas.paths import PROCESSED, ensure_dirs

logger = logging.getLogger(__name__)

AA_SHELL = list("ACDEFGHIKLMNPQRSTVWY")


class DesignInvariantError(AssertionError):
    """Raised when a design violates fixed-catalytic or redesignable-only rules."""


def catalytic_seq_indices(pocket: dict[str, Any]) -> list[int]:
    idxs = []
    for r in pocket.get("catalytic_residues") or []:
        if r.get("seq_index") is None:
            raise DesignInvariantError(
                f"{pocket.get('enzyme_id')}: catalytic residue missing seq_index"
            )
        idxs.append(int(r["seq_index"]))
    return idxs


def redesignable_seq_indices(pocket: dict[str, Any]) -> set[int]:
    return {
        int(r["seq_index"])
        for r in (pocket.get("redesignable") or [])
        if r.get("seq_index") is not None
    }


def assert_design_invariants(
    designed_seq: str,
    wt_seq: str,
    pocket: dict[str, Any],
) -> None:
    """Hard asserts: catalytic identity + mutations ⊆ redesignable positions."""
    if len(designed_seq) != len(wt_seq):
        raise DesignInvariantError(
            f"{pocket.get('enzyme_id')}: length mismatch "
            f"designed={len(designed_seq)} wt={len(wt_seq)}"
        )

    for idx in catalytic_seq_indices(pocket):
        if designed_seq[idx] != wt_seq[idx]:
            raise DesignInvariantError(
                f"{pocket.get('enzyme_id')}: catalytic position {idx} "
                f"changed {wt_seq[idx]}→{designed_seq[idx]}"
            )

    redesignable = redesignable_seq_indices(pocket)
    mutated = {
        i for i, (a, b) in enumerate(zip(designed_seq, wt_seq, strict=True)) if a != b
    }
    illegal = mutated - redesignable
    if illegal:
        raise DesignInvariantError(
            f"{pocket.get('enzyme_id')}: mutations outside redesignable set: "
            f"{sorted(illegal)[:20]}"
        )
    # Explicit set form from the plan.
    assert mutated <= redesignable


def mutate_shell(
    wt_seq: str,
    pocket: dict[str, Any],
    *,
    n_mutations: int = 3,
    rng: np.random.Generator | None = None,
) -> str:
    """Create a valid shell-only design (for demos / CI when MPNN is unavailable)."""
    rng = rng or np.random.default_rng(0)
    redesignable = sorted(redesignable_seq_indices(pocket))
    if not redesignable:
        raise ValueError(f"{pocket.get('enzyme_id')}: no redesignable positions")
    n_mutations = min(n_mutations, len(redesignable))
    positions = rng.choice(redesignable, size=n_mutations, replace=False)
    seq = list(wt_seq)
    for pos in positions:
        choices = [aa for aa in AA_SHELL if aa != seq[pos]]
        seq[pos] = str(rng.choice(choices))
    designed = "".join(seq)
    assert_design_invariants(designed, wt_seq, pocket)
    return designed


def generate_mock_designs(
    pocket: dict[str, Any],
    *,
    n_sequences: int = 100,
    seed: int = 7,
) -> list[dict[str, str]]:
    """Shell-only mock designs for offline / CI pipelines."""
    rng = np.random.default_rng(seed)
    wt = pocket["sequence"]
    eid = pocket["enzyme_id"]
    records = []
    for i in range(n_sequences):
        n_mut = int(rng.integers(1, min(6, max(2, len(redesignable_seq_indices(pocket))))))
        seq = mutate_shell(wt, pocket, n_mutations=n_mut, rng=rng)
        records.append(
            {
                "enzyme_id": eid,
                "design_id": f"{eid}_mock_{i:04d}",
                "sequence": seq,
                "generator": "mock_shell",
            }
        )
    return records


def validate_records(
    records: list[dict[str, str]],
    pocket: dict[str, Any],
) -> list[dict[str, str]]:
    wt = pocket["sequence"]
    eid = pocket["enzyme_id"]
    out = []
    for rec in records:
        seq = rec["sequence"]
        if rec.get("enzyme_id") and rec["enzyme_id"] not in {"", eid}:
            # Allow blank enzyme_id from FASTA; otherwise must match.
            if rec["enzyme_id"] != eid:
                raise DesignInvariantError(
                    f"FASTA enzyme_id {rec['enzyme_id']} != pocket {eid}"
                )
        assert_design_invariants(seq, wt, pocket)
        out.append(
            {
                "enzyme_id": eid,
                "design_id": rec.get("design_id") or f"{eid}_design",
                "sequence": seq,
                "generator": rec.get("generator") or "imported",
            }
        )
    return out


def run_generate(
    enzyme_ids: list[str],
    *,
    n_sequences: int = 100,
    from_sequences: Path | None = None,
    use_mock: bool = False,
    seed: int = 7,
    export_jobs: bool = True,
) -> pd.DataFrame:
    """Generate or import designs for panel enzymes; persist designs parquet + FASTA."""
    ensure_dirs()
    # Ensure pocket JSONs exist.
    run_pockets(enzyme_ids=enzyme_ids)

    all_rows: list[dict[str, Any]] = []
    for eid in enzyme_ids:
        pocket = load_pocket(eid)
        if export_jobs:
            export_mpnn_job(pocket, n_sequences=n_sequences)

        if from_sequences is not None:
            # FASTA may contain multiple enzymes; filter to this eid.
            parsed = parse_design_fasta(Path(from_sequences))
            mine = [
                r
                for r in parsed
                if (not r.get("enzyme_id")) or r["enzyme_id"] == eid
            ]
            # If headers lack enzyme_id, treat whole file as this enzyme only when
            # generating for a single enzyme.
            if not mine and len(enzyme_ids) == 1:
                mine = [{**r, "enzyme_id": eid} for r in parsed]
            records = validate_records(mine, pocket)
        elif use_mock:
            # Process-stable seed offset (avoid Python's salted hash()).
            offset = sum(ord(c) for c in eid) % 10_000
            records = generate_mock_designs(
                pocket, n_sequences=n_sequences, seed=seed + offset
            )
        else:
            raise FileNotFoundError(
                "No designs provided. Pass --from-sequences PATH or --mock "
                "(ProteinMPNN is an external runner: export mpnn_jobs/, import FASTA)."
            )

        for rec in records:
            muts = [
                f"{wt}{i+1}{des}"
                for i, (wt, des) in enumerate(
                    zip(pocket["sequence"], rec["sequence"], strict=True)
                )
                if wt != des
            ]
            all_rows.append(
                {
                    **rec,
                    "n_mutations": len(muts),
                    "mutations": ",".join(muts),
                }
            )

    df = pd.DataFrame(all_rows)
    out_dir = PROCESSED / "design"
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "designs.parquet"
    df.to_parquet(parquet_path, index=False)
    write_design_fasta(
        [{"enzyme_id": r["enzyme_id"], "design_id": r["design_id"], "sequence": r["sequence"]}
         for r in all_rows],
        out_dir / "designs.fasta",
    )
    (out_dir / "generate_meta.json").write_text(
        json.dumps(
            {
                "n_enzymes": len(enzyme_ids),
                "n_designs": len(df),
                "n_sequences_requested": n_sequences,
                "source": "fasta" if from_sequences else ("mock" if use_mock else "unknown"),
            },
            indent=2,
        )
    )
    logger.info("Generated %d designs for %d enzymes → %s", len(df), len(enzyme_ids), parquet_path)
    return df
