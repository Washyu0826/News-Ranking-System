"""Embedding-based candidate generation with sklearn fallback."""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingCandidateGenerator:
    """Retrieve candidates via TF-IDF cosine similarity."""

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(max_features=2000)
        self.news_ids: list[str] = []
        self.matrix = None

    def fit(self, news_df: pd.DataFrame) -> None:
        corpus = (news_df["title"].fillna("") + " " + news_df["abstract"].fillna("")).tolist()
        self.news_ids = news_df["news_id"].tolist()
        self.matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query_text: str, top_n: int = 50) -> list[str]:
        if self.matrix is None:
            return []
        q = self.vectorizer.transform([query_text])
        sims = cosine_similarity(q, self.matrix).ravel()
        idx = sims.argsort()[::-1][:top_n]
        return [self.news_ids[i] for i in idx]
