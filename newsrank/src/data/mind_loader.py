"""MIND dataset loader and toy dataset utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

NEWS_COLUMNS = [
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]
BEHAVIOR_COLUMNS = ["impression_id", "user_id", "time", "history", "impressions"]


@dataclass
class MindDataset:
    """Container for MIND dataset tables."""

    news: pd.DataFrame
    behaviors: pd.DataFrame
    impressions: pd.DataFrame


def load_mind_dataset(news_path: str, behaviors_path: str) -> MindDataset:
    """Load MIND files and convert impressions into a training dataframe."""
    n_path, b_path = Path(news_path), Path(behaviors_path)
    if not n_path.exists() or not b_path.exists():
        raise FileNotFoundError(
            f"Expected files at news={n_path.resolve()} and behaviors={b_path.resolve()}"
        )

    news = pd.read_csv(n_path, sep="\t", header=None, names=NEWS_COLUMNS)
    behaviors = pd.read_csv(b_path, sep="\t", header=None, names=BEHAVIOR_COLUMNS)
    impressions = expand_impressions(behaviors)
    return MindDataset(news=news, behaviors=behaviors, impressions=impressions)


def expand_impressions(behaviors: pd.DataFrame) -> pd.DataFrame:
    """Convert impression text into row-wise labeled interactions."""
    rows: list[dict[str, object]] = []
    for _, row in behaviors.iterrows():
        raw = str(row["impressions"]).split()
        for token in raw:
            news_id, label = token.split("-")
            rows.append(
                {
                    "impression_id": row["impression_id"],
                    "user_id": row["user_id"],
                    "news_id": news_id,
                    "label": int(label),
                    "timestamp": row["time"],
                    "candidate_group_id": row["impression_id"],
                }
            )
    return pd.DataFrame(rows)


def generate_toy_dataset() -> MindDataset:
    """Create a tiny in-memory dataset for MVP runs without MIND files."""
    news = pd.DataFrame(
        [
            ["N1", "Tech", "AI", "AI beats benchmark", "Model news", "", "", ""],
            ["N2", "Sports", "NBA", "Team wins finals", "Sports update", "", "", ""],
            ["N3", "Tech", "Gadgets", "New phone released", "Device launch", "", "", ""],
            ["N4", "World", "Politics", "Election updates", "Global outlook", "", "", ""],
        ],
        columns=NEWS_COLUMNS,
    )
    behaviors = pd.DataFrame(
        [
            [1, "U1", "2024-01-01 10:00:00", "N1 N3", "N1-1 N2-0 N4-0"],
            [2, "U2", "2024-01-02 10:00:00", "N2", "N2-1 N3-0 N4-0"],
        ],
        columns=BEHAVIOR_COLUMNS,
    )
    return MindDataset(news=news, behaviors=behaviors, impressions=expand_impressions(behaviors))
