from src.features.ranking_features import token_overlap
from src.features.user_features import category_preference_score


def test_category_preference_score() -> None:
    profile = {"category_preference": {"Tech": 0.6, "Sports": 0.4}}
    assert category_preference_score(profile, "Tech") == 0.6


def test_title_overlap_score() -> None:
    assert token_overlap("ai model beats", "ai model") > 0.5
