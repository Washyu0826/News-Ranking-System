"""User-news interaction feature builders."""

from __future__ import annotations

from typing import Any


def token_overlap(a: str, b: str) -> float:
    """Jaccard overlap between title tokens."""
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def build_interaction_features(user_profile: dict[str, Any], article: dict[str, Any]) -> dict[str, float]:
    """Create core interaction features for ranking."""
    top_cats = set(user_profile.get("user_top_categories", []))
    match = 1.0 if article.get("category") in top_cats else 0.0
    return {
        "category_match": match,
        "subcategory_match": 0.0,
        "title_overlap_score": token_overlap(" ".join(user_profile.get("user_history", [])), article.get("title", "")),
        "user_news_embedding_similarity": match,
        "popularity_score": float(article.get("popularity_count", 0.0)),
        "freshness_score": float(article.get("freshness_score", 1.0)),
        "user_activity_score": float(user_profile.get("user_history_length", 0) > 0),
    }
