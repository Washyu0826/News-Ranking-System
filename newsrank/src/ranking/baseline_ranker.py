"""Baseline rule-based ranking model."""

from __future__ import annotations

from typing import Any


class BaselineRanker:
    """Weighted linear scorer for relevance."""

    def score(self, feats: dict[str, float]) -> float:
        return (
            0.35 * feats.get("category_preference_score", 0.0)
            + 0.25 * feats.get("user_news_embedding_similarity", 0.0)
            + 0.20 * feats.get("popularity_score", 0.0)
            + 0.20 * feats.get("freshness_score", 0.0)
        )

    def rank(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for item in items:
            item["score"] = self.score(item["features"])
        return sorted(items, key=lambda x: x["score"], reverse=True)
