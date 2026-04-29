from src.evaluation.metrics import ctr_at_k_simulation, mrr_at_k, ndcg_at_k, recall_at_k


def test_ndcg_at_k() -> None:
    assert ndcg_at_k([[1, 0, 0]], [[0.9, 0.2, 0.1]], 3) > 0.99


def test_mrr_at_k() -> None:
    assert mrr_at_k([[0, 1, 0]], [[0.8, 0.7, 0.6]], 3) == 0.5


def test_recall_at_k() -> None:
    assert recall_at_k([[1, 0, 1]], [[0.9, 0.1, 0.8]], 2) == 1.0


def test_ctr_at_k_simulation() -> None:
    assert ctr_at_k_simulation([[0, 1]], [[0.2, 0.9]], 1) == 1.0
