"""Popularity-based candidate generation."""

from __future__ import annotations

import pandas as pd


class PopularityCandidateGenerator:
    """Retrieve top-N popular news items."""

    def __init__(self, popularity_map: dict[str, int]) -> None:
        self.popularity_map = popularity_map

    def generate(self, news_df: pd.DataFrame, top_n: int = 50, category: str | None = None) -> list[str]:
        df = news_df.copy()
        if category:
            df = df[df["category"] == category]
        df["popularity"] = df["news_id"].map(self.popularity_map).fillna(0)
        return df.sort_values("popularity", ascending=False)["news_id"].head(top_n).tolist()
