"""LightGBM ranker wrapper with graceful fallback."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from lightgbm import LGBMRanker
except Exception:
    LGBMRanker = None


class LGBMRankerModel:
    """Ranking model interface with optional LightGBM backend."""

    def __init__(self) -> None:
        self.model = LGBMRanker(objective="lambdarank", metric="ndcg") if LGBMRanker else None

    def fit(self, x: np.ndarray, y: np.ndarray, group: list[int]) -> None:
        if self.model is None:
            return
        self.model.fit(x, y, group=group)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.zeros(len(x))
        return self.model.predict(x)
