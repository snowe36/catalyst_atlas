"""Frozen ESM-2 sequence embeddings as a chemistry-transfer *control*.

Not the headline — answers: does a protein LM already encode catalytic chemistry?
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from catalyst_atlas.models.device import get_device, require_torch
from catalyst_atlas.paths import PROCESSED, ensure_dirs

logger = logging.getLogger(__name__)

# Small enough for a control; not a 650M flex.
DEFAULT_ESM_MODEL = "esm2_t12_35M_UR50D"


def _load_esm(model_name: str = DEFAULT_ESM_MODEL):
    """Load frozen ESM-2 via fair-esm if available, else transformers."""
    torch = require_torch()
    try:
        import esm  # type: ignore

        model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        model.eval()
        return "fair-esm", model, alphabet, None
    except Exception as exc:
        logger.info("fair-esm unavailable (%s); trying transformers", exc)

    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Need fair-esm or transformers for ESM embeddings. "
            "pip install -e '.[gpu]'"
        ) from exc

    hf_name = {
        "esm2_t12_35M_UR50D": "facebook/esm2_t12_35M_UR50D",
        "esm2_t30_150M_UR50D": "facebook/esm2_t30_150M_UR50D",
    }.get(model_name, f"facebook/{model_name}")
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    model = AutoModel.from_pretrained(hf_name)
    model.eval()
    return "transformers", model, None, tokenizer


def embed_sequences_esm(
    sequences: list[str],
    model_name: str = DEFAULT_ESM_MODEL,
    batch_size: int = 4,
    device: str | None = None,
    max_length: int = 1022,
) -> np.ndarray:
    torch = require_torch()
    backend, model, alphabet, tokenizer = _load_esm(model_name)
    dev = get_device() if device is None else torch.device(device)
    model = model.to(dev)

    embeddings: list[np.ndarray] = []
    if backend == "fair-esm":
        batch_converter = alphabet.get_batch_converter()
        repr_layer = model.num_layers
        for start in range(0, len(sequences), batch_size):
            chunk = sequences[start : start + batch_size]
            data = [(f"s{i}", (seq or "A")[:max_length]) for i, seq in enumerate(chunk)]
            _, _, toks = batch_converter(data)
            toks = toks.to(dev)
            with torch.no_grad():
                out = model(toks, repr_layers=[repr_layer], return_contacts=False)
            reps = out["representations"][repr_layer]
            for i, seq in enumerate(chunk):
                # skip BOS/EOS — tokens 1..L
                L = min(len(seq or "A"), max_length)
                emb = reps[i, 1 : L + 1].mean(dim=0).cpu().numpy()
                embeddings.append(emb)
    else:
        assert tokenizer is not None
        for start in range(0, len(sequences), batch_size):
            chunk = [(seq or "A")[:max_length] for seq in sequences[start : start + batch_size]]
            enc = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length + 2,
            )
            enc = {k: v.to(dev) for k, v in enc.items()}
            with torch.no_grad():
                out = model(**enc)
            hidden = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)
            # mean pool excluding padding
            summed = (hidden * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1)
            emb = (summed / denom).cpu().numpy()
            for row in emb:
                embeddings.append(row)

    return np.stack(embeddings, axis=0).astype(np.float32)


def run_esm_embed(
    model_name: str = DEFAULT_ESM_MODEL,
    batch_size: int = 4,
    device: str | None = None,
) -> dict[str, Any]:
    ensure_dirs()
    meta_path = PROCESSED / "features_full_meta.parquet"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}; run cat-embed first")
    meta = pd.read_parquet(meta_path).reset_index(drop=True)
    if "sequence" not in meta.columns:
        raise RuntimeError("feature meta lacks sequence column")
    seqs = meta["sequence"].fillna("").astype(str).tolist()
    logger.info("Encoding %d sequences with frozen %s", len(seqs), model_name)
    X = embed_sequences_esm(
        seqs, model_name=model_name, batch_size=batch_size, device=device
    )
    np.save(PROCESSED / "embedding_esm.npy", X)
    meta.to_parquet(PROCESSED / "embedding_esm_meta.parquet", index=False)
    summary = {
        "n_enzymes": int(X.shape[0]),
        "dim": int(X.shape[1]),
        "model": model_name,
        "path": str(PROCESSED / "embedding_esm.npy"),
    }
    logger.info("Wrote ESM embeddings %s", summary)
    return summary
