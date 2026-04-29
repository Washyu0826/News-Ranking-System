"""Data preprocessing helpers."""

from __future__ import annotations

import pandas as pd


def normalize_text_columns(news_df: pd.DataFrame) -> pd.DataFrame:
    """Fill nulls in textual columns for robust downstream feature pipelines."""
    out = news_df.copy()
    for col in ["title", "abstract", "category", "subcategory"]:
        out[col] = out[col].fillna("")
    return out
