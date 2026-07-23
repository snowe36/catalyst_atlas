"""Resolve the primary chemistry label column for evaluation."""

from __future__ import annotations

import pandas as pd


def chemistry_label_col(meta: pd.DataFrame) -> str:
    """Prefer chemistry_family; fall back to chemistry_class."""
    if "chemistry_family" in meta.columns and meta["chemistry_family"].notna().any():
        return "chemistry_family"
    return "chemistry_class"
