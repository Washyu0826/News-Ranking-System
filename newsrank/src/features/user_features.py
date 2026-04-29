"""User-level feature engineering."""

from __future__ import annotations

from collections import Counter

import pandas as pd


def build_user_profiles(behaviors_df: pd.DataFrame, news_df: pd.DataFrame) -> dict[str, dict[str, object]]:
    """Build simple user profiles from history and clicked logs."""
    news_cat = news_df.set_index("news_id")["category"].to_dict()
    profiles: dict[str, dict[str, object]] = {}
    for _, row in behaviors_df.iterrows():
        user = row["user_id"]
        history = str(row["history"]).split() if pd.notna(row["history"]) else []
        cats = [news_cat.get(n, "Unknown") for n in history]
        cnt = Counter(cats)
        total = sum(cnt.values()) or 1
        profiles[user] = {
            "user_history": history,
            "user_history_length": len(history),
            "user_click_count": len(history),
            "user_top_categories": [c for c, _ in cnt.most_common(3)],
            "category_preference": {k: v / total for k, v in cnt.items()},
            "recent_clicked_news_count": min(len(history), 5),
        }
    return profiles


def category_preference_score(user_profile: dict[str, object], category: str) -> float:
    """Compute user preference score for a category."""
    pref = user_profile.get("category_preference", {})
    return float(pref.get(category, 0.0))
