from src.reranking.diversity_reranker import rerank_with_diversity


def test_diversity_reranking_reduces_duplicate_categories() -> None:
    items = [
        {"news_id": "N1", "category": "Tech", "score": 0.9, "freshness_score": 1.0},
        {"news_id": "N2", "category": "Tech", "score": 0.89, "freshness_score": 1.0},
        {"news_id": "N3", "category": "Sports", "score": 0.88, "freshness_score": 1.0},
    ]
    reranked = rerank_with_diversity(items, top_k=3)
    assert reranked[1]["category"] == "Sports"
