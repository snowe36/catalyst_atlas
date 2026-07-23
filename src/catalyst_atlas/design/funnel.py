"""Chemistry-constrained design funnel: 1000 seq → ~100 AF structures.

Industrial-style campaign:
  generate → hard filters → cheap rank (ESM + fixed-backbone chemistry)
  → top-k/enzyme → AF queue → mechanistic ranking
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd

from catalyst_atlas.design.generate import (
    DesignInvariantError,
    assert_design_invariants,
    redesignable_seq_indices,
)
from catalyst_atlas.design.mpnn import write_design_fasta
from catalyst_atlas.design.pocket import load_pocket
from catalyst_atlas.design.score import embed_sequences, esm_plausibility
from catalyst_atlas.paths import PROCESSED, ensure_dirs

logger = logging.getLogger(__name__)

NEG = set("DE")
POS = set("KRH")
AA_OK = set("ACDEFGHIKLMNPQRSTVWY")

# Pre-AF cheap rank (no structure prediction).
W_CHEAP_ESM = 0.5
W_CHEAP_CHEM = 0.5

DEFAULT_TOP_K = 10
DEFAULT_MAX_MUTATIONS = 40  # shell redesign can touch many of 8–40 redesignable sites


def _charge(aa: str) -> int:
    if aa in NEG:
        return -1
    if aa in POS:
        return 1
    return 0


def _metal_contact_resnums(pocket: dict[str, Any]) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    for c in pocket.get("ligand_contacts") or []:
        kind = str(c.get("ligand_kind") or "").lower()
        name = str(c.get("ligand") or "")
        if kind == "metal" or name in {"Zn", "Fe", "Mg", "Mn", "Ca", "Cu", "Co", "Ni"}:
            keys.add((str(c.get("chain") or "A"), int(c["resnum"])))
    return keys


def fixed_backbone_chemistry(
    designed_seq: str,
    wt_seq: str,
    pocket: dict[str, Any],
) -> float:
    """Pre-AF chemical-environment proxy on the fixed backbone.

    Penalizes charge flips at metal-contact / first-shell positions and large
    property changes in the redesignable shell. Returns [0, 1] (1 = gentle).
    """
    by_idx = {
        int(r["seq_index"]): r
        for r in (pocket.get("redesignable") or [])
        if r.get("seq_index") is not None
    }
    metal_keys = _metal_contact_resnums(pocket)
    penalties: list[float] = []
    n_mut = 0
    for idx, r in by_idx.items():
        if idx >= len(designed_seq) or idx >= len(wt_seq):
            continue
        wt_aa, des_aa = wt_seq[idx], designed_seq[idx]
        if wt_aa == des_aa:
            continue
        n_mut += 1
        pen = 0.05  # base edit cost
        if _charge(wt_aa) != _charge(des_aa):
            pen += 0.25
            key = (str(r.get("chain") or "A"), int(r["resnum"]))
            if key in metal_keys or r.get("shell") == "first":
                pen += 0.35  # charge flip near metal / first shell
        if (wt_aa in "STNQ" and des_aa in "AILMFVW") or (
            wt_aa in "AILMFVW" and des_aa in "STNQ"
        ):
            pen += 0.1
        penalties.append(min(pen, 1.0))

    if n_mut == 0:
        return 1.0
    mean_pen = float(np.mean(penalties)) if penalties else 0.0
    # Mild global mutation burden.
    burden = min(1.0, n_mut / max(int(pocket.get("n_redesignable") or 1), 1))
    return float(max(0.0, 1.0 - 0.7 * mean_pen - 0.3 * burden))


def hard_filter_design(
    rec: dict[str, Any] | pd.Series,
    pocket: dict[str, Any],
    *,
    max_mutations: int = DEFAULT_MAX_MUTATIONS,
) -> tuple[bool, str]:
    """Return (ok, reason). Fail closed on invariant / sanity breaks."""
    seq = str(rec["sequence"]).upper().replace(" ", "")
    wt = pocket["sequence"]
    if not seq or any(c not in AA_OK for c in seq):
        return False, "broken_sequence"
    if len(seq) != len(wt):
        return False, "length_mismatch"
    try:
        assert_design_invariants(seq, wt, pocket)
    except (DesignInvariantError, AssertionError) as exc:
        return False, f"invariant:{exc}"
    n_mut = sum(a != b for a, b in zip(seq, wt, strict=True))
    if n_mut == 0:
        return False, "no_mutations"
    if n_mut > max_mutations:
        return False, "too_many_mutations"
    # Extra sanity: mutations must be in redesignable (already asserted).
    _ = redesignable_seq_indices(pocket)
    return True, "ok"


def cheap_rank_score(
    *,
    esm: float,
    chemistry: float,
) -> float:
    return float(W_CHEAP_ESM * esm + W_CHEAP_CHEM * chemistry)


def run_funnel(
    designs: pd.DataFrame | None = None,
    *,
    top_k: int = DEFAULT_TOP_K,
    max_mutations: int = DEFAULT_MAX_MUTATIONS,
    enzyme_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Filter + cheap-rank all designs; write AF queue (WT + top-k/enzyme)."""
    ensure_dirs()
    if designs is None:
        path = PROCESSED / "design" / "designs.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run cat-design-generate first")
        designs = pd.read_parquet(path)

    if enzyme_ids is None:
        enzyme_ids = sorted(designs["enzyme_id"].unique().tolist())

    filter_rows: list[dict[str, Any]] = []
    af_queue: list[dict[str, Any]] = []

    for eid in enzyme_ids:
        pocket = load_pocket(eid)
        wt = pocket["sequence"]
        wt_emb = embed_sequences([wt])[0]

        # Round 0: always queue WT for AF baseline.
        af_queue.append(
            {
                "enzyme_id": eid,
                "design_id": "WT",
                "sequence": wt,
                "is_wt": True,
                "cheap_rank_score": 1.0,
                "stage": "wt_baseline",
            }
        )

        subset = designs[designs["enzyme_id"] == eid]
        ranked: list[dict[str, Any]] = []
        for _, rec in subset.iterrows():
            ok, reason = hard_filter_design(rec, pocket, max_mutations=max_mutations)
            base = {
                "enzyme_id": eid,
                "design_id": str(rec["design_id"]),
                "sequence": str(rec["sequence"]),
                "n_mutations": int(rec.get("n_mutations") or 0),
                "mutations": rec.get("mutations", ""),
                "passed_hard_filter": ok,
                "filter_reason": reason,
            }
            if not ok:
                filter_rows.append({**base, "esm_plausibility": None, "fixed_backbone_chemistry": None, "cheap_rank_score": None})
                continue
            seq = str(rec["sequence"])
            esm = esm_plausibility(seq, wt, wt_emb=wt_emb)
            chem = fixed_backbone_chemistry(seq, wt, pocket)
            cheap = cheap_rank_score(esm=esm, chemistry=chem)
            row = {
                **base,
                "esm_plausibility": esm,
                "fixed_backbone_chemistry": chem,
                "cheap_rank_score": cheap,
            }
            filter_rows.append(row)
            ranked.append(row)

        ranked.sort(key=lambda r: r["cheap_rank_score"], reverse=True)
        for r in ranked[:top_k]:
            af_queue.append(
                {
                    "enzyme_id": r["enzyme_id"],
                    "design_id": r["design_id"],
                    "sequence": r["sequence"],
                    "is_wt": False,
                    "cheap_rank_score": r["cheap_rank_score"],
                    "esm_plausibility": r["esm_plausibility"],
                    "fixed_backbone_chemistry": r["fixed_backbone_chemistry"],
                    "n_mutations": r["n_mutations"],
                    "stage": "af_shortlist",
                }
            )

    filt_df = pd.DataFrame(filter_rows)
    queue_df = pd.DataFrame(af_queue)
    out_dir = PROCESSED / "design"
    out_dir.mkdir(parents=True, exist_ok=True)
    filt_path = out_dir / "funnel_filter.parquet"
    queue_path = out_dir / "af_queue.parquet"
    filt_df.to_parquet(filt_path, index=False)
    queue_df.to_parquet(queue_path, index=False)

    fasta_recs = [
        {
            "enzyme_id": r["enzyme_id"],
            "design_id": r["design_id"],
            "sequence": r["sequence"],
        }
        for r in af_queue
    ]
    fasta_path = write_design_fasta(fasta_recs, out_dir / "af_queue.fasta")

    n_in = int(len(designs))
    n_pass = int(filt_df["passed_hard_filter"].sum()) if len(filt_df) else 0
    n_af = int((~queue_df["is_wt"]).sum()) if len(queue_df) else 0
    n_wt = int(queue_df["is_wt"].sum()) if len(queue_df) else 0
    meta = {
        "n_input_designs": n_in,
        "n_passed_hard_filter": n_pass,
        "n_af_designs": n_af,
        "n_af_wt": n_wt,
        "n_af_total": n_af + n_wt,
        "top_k": top_k,
        "max_mutations": max_mutations,
        "cheap_weights": {"esm": W_CHEAP_ESM, "chemistry": W_CHEAP_CHEM},
        "story": (
            "Chemistry-constrained funnel: generative candidates → hard filters → "
            "ESM + fixed-backbone chemistry → AF shortlist (~50–100), not 1000 AF jobs."
        ),
        "paths": {
            "filter": str(filt_path),
            "af_queue": str(queue_path),
            "af_fasta": str(fasta_path),
        },
    }
    (out_dir / "funnel_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info(
        "Funnel: %d designs → %d hard-pass → %d AF designs + %d WT",
        n_in,
        n_pass,
        n_af,
        n_wt,
    )
    return meta


def load_af_queue() -> pd.DataFrame:
    path = PROCESSED / "design" / "af_queue.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run cat-design-funnel first")
    return pd.read_parquet(path)
