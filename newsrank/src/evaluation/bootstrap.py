"""Bootstrap confidence interval utilities."""

from __future__ import annotations

import numpy as np


def bootstrap_metric_diff(metric_values_a: list[float], metric_values_b: list[float], n_bootstrap: int = 1000) -> tuple[float, float, float]:
    arr_a, arr_b = np.array(metric_values_a), np.array(metric_values_b)
    n = min(len(arr_a), len(arr_b))
    if n == 0:
        return 0.0, 0.0, 0.0
    diffs = []
    for _ in range(n_bootstrap):
        idx = np.random.randint(0, n, size=n)
        diffs.append(float(np.mean(arr_b[idx] - arr_a[idx])))
    return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
