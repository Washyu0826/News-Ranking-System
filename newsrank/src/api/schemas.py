"""Pydantic schemas for NewsRank API."""

from __future__ import annotations

from pydantic import BaseModel


class RankRequest(BaseModel):
    user_id: str
    candidate_news_ids: list[str]
    top_k: int = 10
    use_diversity_reranking: bool = True


class RankedItem(BaseModel):
    rank: int
    news_id: str
    title: str
    category: str
    score: float
    reason: str


class RankResponse(BaseModel):
    user_id: str
    results: list[RankedItem]
