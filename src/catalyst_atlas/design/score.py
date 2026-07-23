"""chemistry_preservation_score — proxies for keeping mechanistic chemistry, not catalysis."""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd

from catalyst_atlas.design.pocket import load_pocket
from catalyst_atlas.design.predict import (
    load_prediction_metrics,
    mock_prediction_metrics,
    structure_confidence_from_metrics,
)
from catalyst_atlas.paths import PROCESSED, ensure_dirs

logger = logging.getLogger(__name__)

W_GEOMETRY = 0.4
W_STRUCTURE = 0.3
W_ESM = 0.3


def reference_geometry_vector(pocket: dict[str, Any]) -> np.ndarray:
    """Catalytic pairwise distances + ligand-contact distances (atlas-style constraints)."""
    cat = pocket.get("catalytic_residues") or []
    pairs: list[float] = []
    for i in range(len(cat)):
        for j in range(i + 1, len(cat)):
            a = np.array(cat[i]["xyz"], dtype=float)
            b = np.array(cat[j]["xyz"], dtype=float)
            pairs.append(float(np.linalg.norm(a - b)))
    contacts = [float(c["distance"]) for c in (pocket.get("ligand_contacts") or [])]
    # Stable padded vector for RMSE comparison.
    vec = np.array(pairs + contacts, dtype=float)
    if vec.size == 0:
        return np.zeros(1, dtype=float)
    return vec


def geometry_preservation(
    query_vec: np.ndarray,
    reference_vec: np.ndarray,
    *,
    scale_angstrom: float = 2.0,
) -> float:
    """Map distance-vector RMSE to [0, 1] (1 = identical catalytic geometry)."""
    if reference_vec.size == 0 and query_vec.size == 0:
        return 1.0
    n = min(reference_vec.size, query_vec.size)
    if n == 0:
        return 0.0
    rmse = float(np.sqrt(np.mean((query_vec[:n] - reference_vec[:n]) ** 2)))
    return float(np.exp(-rmse / scale_angstrom))


def _kmer_embed(seq: str, k: int = 3) -> np.ndarray:
    aa = "ACDEFGHIKLMNPQRSTVWY"
    index = {c: i for i, c in enumerate(aa)}
    # Compact hashed bag-of-k-mers (CI-safe ESM stand-in).
    vec = np.zeros(256, dtype=float)
    s = (seq or "").upper()
    if len(s) < k:
        return vec
    for i in range(len(s) - k + 1):
        chunk = s[i : i + k]
        if any(c not in index for c in chunk):
            continue
        h = 0
        for c in chunk:
            h = h * 20 + index[c]
        vec[h % 256] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def embed_sequences(sequences: list[str]) -> np.ndarray:
    """ESM-2 when available; otherwise hashed k-mer embeddings for CI."""
    try:
        from catalyst_atlas.models.esm_embed import embed_sequences_esm

        return embed_sequences_esm(sequences)
    except Exception as exc:
        logger.info("ESM unavailable (%s); using k-mer embedding fallback", exc)
        return np.vstack([_kmer_embed(s) for s in sequences])


def esm_plausibility(design_seq: str, wt_seq: str, *, wt_emb: np.ndarray | None = None) -> float:
    """Cosine similarity to WT in embedding space, clipped to [0, 1]."""
    if wt_emb is None:
        embs = embed_sequences([wt_seq, design_seq])
        wt_v, des_v = embs[0], embs[1]
    else:
        des_v = embed_sequences([design_seq])[0]
        wt_v = wt_emb
    denom = (np.linalg.norm(wt_v) * np.linalg.norm(des_v)) + 1e-8
    cos = float(np.dot(wt_v, des_v) / denom)
    return float(max(0.0, min(1.0, cos)))


def chemistry_preservation_score(
    *,
    geometry: float,
    structure: float,
    esm: float,
) -> float:
    """Weighted proxy score — not a measured catalytic rate."""
    return float(
        W_GEOMETRY * geometry + W_STRUCTURE * structure + W_ESM * esm
    )


def _geometry_for_design(
    pocket: dict[str, Any],
    *,
    design_id: str,
    n_mutations: int,
    use_mock_jitter: bool,
    rng: np.random.Generator,
) -> float:
    """Compare design geometry to WT pocket reference.

    Without a predicted structure, ProteinMPNN fixed-backbone implies catalytic
    coordinates unchanged → geometry ≈ 1. In mock mode we add a mild mutation-
    linked penalty so WT-relative figures are not flat.
    """
    ref = reference_geometry_vector(pocket)
    metrics = load_prediction_metrics(pocket["enzyme_id"], design_id)
    if metrics and metrics.get("geometry_vector") is not None:
        query = np.asarray(metrics["geometry_vector"], dtype=float)
        return geometry_preservation(query, ref)

    if use_mock_jitter:
        n_red = max(int(pocket.get("n_redesignable") or 1), 1)
        # Soft penalty: more shell edits → slightly worse proxy geometry.
        noise = 0.15 * (n_mutations / n_red) + 0.02 * float(rng.random())
        query = ref * (1.0 + rng.normal(scale=noise, size=ref.shape))
        return geometry_preservation(query, ref)

    # Fixed-backbone assumption when AF geometry is absent.
    return 1.0


def score_enzyme_designs(
    enzyme_id: str,
    designs: pd.DataFrame,
    *,
    mock_predictions: bool = False,
    seed: int = 7,
) -> pd.DataFrame:
    """Score WT baseline first, then each design; attach WT deltas."""
    pocket = load_pocket(enzyme_id)
    wt_seq = pocket["sequence"]
    rng = np.random.default_rng(seed)

    if mock_predictions:
        mock_prediction_metrics(enzyme_id, "WT", mean_plddt=90.0, pocket_pae=4.0)

    wt_metrics = load_prediction_metrics(enzyme_id, "WT")
    wt_geom = 1.0  # self-reference
    wt_struct = structure_confidence_from_metrics(wt_metrics)
    wt_emb = embed_sequences([wt_seq])[0]
    wt_esm = 1.0
    wt_score = chemistry_preservation_score(
        geometry=wt_geom, structure=wt_struct, esm=wt_esm
    )

    rows: list[dict[str, Any]] = [
        {
            "enzyme_id": enzyme_id,
            "design_id": "WT",
            "is_wt": True,
            "sequence": wt_seq,
            "n_mutations": 0,
            "mutations": "",
            "geometry_preservation": wt_geom,
            "structure_confidence": wt_struct,
            "esm_plausibility": wt_esm,
            "chemistry_preservation_score": wt_score,
            "delta_geometry_vs_wt": 0.0,
            "delta_score_vs_wt": 0.0,
            "chemistry_family": pocket["reaction"]["chemistry_family"],
            "mechanistic_pattern": pocket["reaction"]["mechanistic_pattern"],
        }
    ]

    subset = designs[designs["enzyme_id"] == enzyme_id]
    for _, rec in subset.iterrows():
        design_id = str(rec["design_id"])
        seq = str(rec["sequence"])
        n_mut = int(rec.get("n_mutations") or 0)

        if mock_predictions:
            # Slightly lower / noisier pLDDT than WT for mock designs.
            mock_prediction_metrics(
                enzyme_id,
                design_id,
                mean_plddt=float(rng.uniform(70.0, 92.0)),
                pocket_pae=float(rng.uniform(3.0, 12.0)),
            )

        geom = _geometry_for_design(
            pocket,
            design_id=design_id,
            n_mutations=n_mut,
            use_mock_jitter=mock_predictions,
            rng=rng,
        )
        struct = structure_confidence_from_metrics(
            load_prediction_metrics(enzyme_id, design_id)
        )
        esm = esm_plausibility(seq, wt_seq, wt_emb=wt_emb)
        score = chemistry_preservation_score(geometry=geom, structure=struct, esm=esm)
        rows.append(
            {
                "enzyme_id": enzyme_id,
                "design_id": design_id,
                "is_wt": False,
                "sequence": seq,
                "n_mutations": n_mut,
                "mutations": rec.get("mutations", ""),
                "geometry_preservation": geom,
                "structure_confidence": struct,
                "esm_plausibility": esm,
                "chemistry_preservation_score": score,
                "delta_geometry_vs_wt": geom - wt_geom,
                "delta_score_vs_wt": score - wt_score,
                "chemistry_family": pocket["reaction"]["chemistry_family"],
                "mechanistic_pattern": pocket["reaction"]["mechanistic_pattern"],
            }
        )

    return pd.DataFrame(rows)


def run_score(
    enzyme_ids: list[str] | None = None,
    *,
    mock_predictions: bool = False,
    seed: int = 7,
    af_queue_only: bool = False,
) -> pd.DataFrame:
    """Mechanistic ranking after AF (or mock AF).

    If ``af_queue_only``, score WT + funnel shortlist only (~50–100 designs),
    not the full generative pool.
    """
    ensure_dirs()
    if af_queue_only:
        queue_path = PROCESSED / "design" / "af_queue.parquet"
        if not queue_path.exists():
            raise FileNotFoundError(
                f"Missing {queue_path}; run cat-design-funnel before AF scoring"
            )
        queue = pd.read_parquet(queue_path)
        designs = queue[~queue["is_wt"]][
            [c for c in ("enzyme_id", "design_id", "sequence", "n_mutations", "mutations") if c in queue.columns]
        ].copy()
        if "n_mutations" not in designs.columns:
            designs["n_mutations"] = 0
        if "mutations" not in designs.columns:
            designs["mutations"] = ""
    else:
        designs_path = PROCESSED / "design" / "designs.parquet"
        if not designs_path.exists():
            raise FileNotFoundError(f"Missing {designs_path}; run cat-design-generate first")
        designs = pd.read_parquet(designs_path)

    if enzyme_ids is None:
        enzyme_ids = sorted(designs["enzyme_id"].unique().tolist())

    frames = [
        score_enzyme_designs(
            eid, designs, mock_predictions=mock_predictions, seed=seed
        )
        for eid in enzyme_ids
    ]
    out = pd.concat(frames, ignore_index=True)
    out_path = PROCESSED / "design" / "scores.parquet"
    out.to_parquet(out_path, index=False)
    meta = {
        "n_enzymes": len(enzyme_ids),
        "n_rows": int(len(out)),
        "af_queue_only": af_queue_only,
        "weights": {
            "geometry": W_GEOMETRY,
            "structure": W_STRUCTURE,
            "esm": W_ESM,
        },
        "score_name": "chemistry_preservation_score",
        "note": "Proxies for chemistry preservation — not measured catalysis.",
    }
    (PROCESSED / "design" / "score_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("Scored %d rows → %s", len(out), out_path)
    return out
