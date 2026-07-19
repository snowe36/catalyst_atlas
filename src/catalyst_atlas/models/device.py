"""Optional torch device helpers for the learned / ESM tracks."""

from __future__ import annotations


def require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyTorch is required for the learned / ESM tracks. "
            "Install with: pip install -e '.[gpu]'"
        ) from exc
    return torch


def get_device(prefer_cuda: bool = True):
    torch = require_torch()
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
