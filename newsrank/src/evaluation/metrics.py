"""Evaluation metrics for ranking quality and feed diversity."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from sklearn.metrics import roc_auc_score


def auc_score(y_true: list[int], y_score: list[float]) -> float:
    return float(roc_auc_score(y_true, y_score))


def _dcg(labels: list[int], k: int) -> float:
    return sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(labels[:k]))


def ndcg_at_k(grouped_labels: list[list[int]], grouped_scores: list[list[float]], k: int = 10) -> float:
    vals = []
    for labels, scores in zip(grouped_labels, grouped_scores):
        order = np.argsort(scores)[::-1]
        pred = [labels[i] for i in order]
        ideal = sorted(labels, reverse=True)
        idcg = _dcg(ideal, k)
        vals.append(0.0 if idcg == 0 else _dcg(pred, k) / idcg)
    return float(np.mean(vals)) if vals else 0.0


def mrr_at_k(grouped_labels: list[list[int]], grouped_scores: list[list[float]], k: int = 10) -> float:
    rrs = []
    for labels, scores in zip(grouped_labels, grouped_scores):
        order = np.argsort(scores)[::-1][:k]
        rr = 0.0
        for rank, idx in enumerate(order, 1):
            if labels[idx] == 1:
                rr = 1.0 / rank
                break
        rrs.append(rr)
    return float(np.mean(rrs)) if rrs else 0.0


def recall_at_k(grouped_labels: list[list[int]], grouped_scores: list[list[float]], k: int = 10) -> float:
    vals = []
    for labels, scores in zip(grouped_labels, grouped_scores):
        order = np.argsort(scores)[::-1][:k]
        hit = sum(labels[i] for i in order)
        total = sum(labels)
        vals.append(0.0 if total == 0 else hit / total)
    return float(np.mean(vals)) if vals else 0.0


def precision_at_k(grouped_labels: list[list[int]], grouped_scores: list[list[float]], k: int = 10) -> float:
    vals = []
    for labels, scores in zip(grouped_labels, grouped_scores):
        order = np.argsort(scores)[::-1][:k]
        vals.append(sum(labels[i] for i in order) / max(k, 1))
    return float(np.mean(vals)) if vals else 0.0


def ctr_at_k_simulation(grouped_labels: list[list[int]], grouped_scores: list[list[float]], k: int = 10) -> float:
    hits = []
    for labels, scores in zip(grouped_labels, grouped_scores):
        order = np.argsort(scores)[::-1][:k]
        hits.append(1.0 if any(labels[i] == 1 for i in order) else 0.0)
    return float(np.mean(hits)) if hits else 0.0


def diversity_at_k(grouped_categories: list[list[str]], k: int = 10) -> float:
    vals = [len(set(cats[:k])) / max(min(len(cats), k), 1) for cats in grouped_categories]
    return float(np.mean(vals)) if vals else 0.0


def coverage_at_k(recommended_ids: Iterable[str], all_news_ids: Iterable[str]) -> float:
    rec, all_ids = set(recommended_ids), set(all_news_ids)
    return 0.0 if not all_ids else len(rec) / len(all_ids)
