"""Offline A/B testing simulation over impression groups."""

from __future__ import annotations

from src.evaluation.bootstrap import bootstrap_metric_diff


def relative_lift(metric_a: float, metric_b: float) -> float:
    """Calculate relative lift percentage."""
    if metric_a == 0:
        return 0.0
    return (metric_b - metric_a) / metric_a * 100.0


def summarize_experiment(metric_values_a: list[float], metric_values_b: list[float]) -> dict[str, float]:
    """Summarize A/B metrics and bootstrap CI."""
    mean_diff, lo, hi = bootstrap_metric_diff(metric_values_a, metric_values_b)
    return {"mean_diff": mean_diff, "ci_lower": lo, "ci_upper": hi}
