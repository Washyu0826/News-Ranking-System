"""FastAPI service for ranking requests."""

from __future__ import annotations

from fastapi import FastAPI

from src.api.schemas import RankRequest, RankResponse, RankedItem
from src.data.mind_loader import generate_toy_dataset
from src.features.article_features import build_article_features
from src.features.ranking_features import build_interaction_features
from src.features.user_features import build_user_profiles, category_preference_score
from src.ranking.baseline_ranker import BaselineRanker
from src.reranking.diversity_reranker import rerank_with_diversity

app = FastAPI(title="NewsRank")

dataset = generate_toy_dataset()
profiles = build_user_profiles(dataset.behaviors, dataset.news)
popularity = dataset.impressions.groupby("news_id")["label"].sum().to_dict()
article_df = build_article_features(dataset.news, popularity)
article_map = article_df.set_index("news_id").to_dict("index")
ranker = BaselineRanker()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/rank", response_model=RankResponse)
def rank(req: RankRequest) -> RankResponse:
    user = profiles.get(req.user_id, {"user_top_categories": [], "category_preference": {}, "user_history": []})
    items = []
    for nid in req.candidate_news_ids:
        if nid not in article_map:
            continue
        article = {"news_id": nid, **article_map[nid]}
        feats = build_interaction_features(user, article)
        feats["category_preference_score"] = category_preference_score(user, article["category"])
        items.append({"news_id": nid, "title": article["title"], "category": article["category"], "freshness_score": article.get("freshness_score", 1.0), "features": feats})
    ranked = ranker.rank(items)
    if req.use_diversity_reranking:
        ranked = rerank_with_diversity(ranked, top_k=req.top_k)
    results = [
        RankedItem(rank=i + 1, news_id=r["news_id"], title=r["title"], category=r["category"], score=float(r.get("final_score", r["score"])), reason=f"Matched interest in {r['category']} with freshness/diversity balancing.")
        for i, r in enumerate(ranked[: req.top_k])
    ]
    return RankResponse(user_id=req.user_id, results=results)


@app.post("/recommend", response_model=RankResponse)
def recommend(req: RankRequest) -> RankResponse:
    return rank(req)


@app.get("/metrics")
def metrics() -> dict[str, str]:
    return {"message": "Offline metrics available via evaluation scripts."}
