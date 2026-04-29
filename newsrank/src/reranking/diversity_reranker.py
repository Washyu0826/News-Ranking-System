"""Diversity-aware reranking module."""

from __future__ import annotations


def rerank_with_diversity(ranked_items: list[dict], top_k: int = 10) -> list[dict]:
    """Apply duplicate category/source penalties while preserving relevance."""
    selected: list[dict] = []
    seen_cat: dict[str, int] = {}
    seen_source: dict[str, int] = {}
    for item in ranked_items:
        cat = item.get("category", "Unknown")
        source = item.get("source", "Unknown")
        penalty = 0.08 * seen_cat.get(cat, 0) + 0.05 * seen_source.get(source, 0)
        bonus = 0.05 * item.get("freshness_score", 1.0)
        item = {**item, "diversity_penalty": penalty, "final_score": item["score"] + bonus - penalty}
        selected.append(item)
        seen_cat[cat] = seen_cat.get(cat, 0) + 1
        seen_source[source] = seen_source.get(source, 0) + 1
    return sorted(selected, key=lambda x: x["final_score"], reverse=True)[:top_k]
