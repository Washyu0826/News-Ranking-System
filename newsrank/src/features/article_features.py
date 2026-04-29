"""Article-level feature engineering."""

from __future__ import annotations

import pandas as pd


def build_article_features(news_df: pd.DataFrame, popularity: dict[str, int] | None = None) -> pd.DataFrame:
    """Create simple article features for MVP ranking."""
    popularity = popularity or {}
    df = news_df.copy()
    df["title_length"] = df["title"].fillna("").str.len()
    df["abstract_length"] = df["abstract"].fillna("").str.len()
    df["has_abstract"] = (df["abstract"].fillna("").str.len() > 0).astype(int)
    df["freshness_score"] = 1.0
    df["popularity_count"] = df["news_id"].map(popularity).fillna(0).astype(float)
    return df
