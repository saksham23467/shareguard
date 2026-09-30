"""Learn-then-Test threshold selection with a Clopper-Pearson bound.

Input: per-secret records from the CALIBRATION split, each with the minimum OCR
confidence among its renderings that were not caught by a pattern rule (i.e. the
confidence the unblur rule sees). A secret is missed at threshold tau if that
confidence >= tau. Thresholds are tested in a fixed order from strictest to
loosest; we stop at the first whose 95% upper bound exceeds alpha. The guarantee
holds only for screens drawn like the calibration screens.
"""
from __future__ import annotations

import math


def _binom_cdf(k: int, n: int, p: float) -> float:
    if p <= 0:
        return 1.0
    if p >= 1:
        return 1.0 if k >= n else 0.0
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def clopper_pearson_upper(k: int, n: int, delta: float = 0.05) -> float:
    """One-sided (1-delta) upper bound on a binomial rate with k events in n."""
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    lo, hi = k / n, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if _binom_cdf(k, n, mid) > delta:
            lo = mid
        else:
            hi = mid
    return hi


def learn_then_test(uncaught_conf: list[float], n_secrets: int, alpha: float,
                    grid: list[float] | None = None, delta: float = 0.05) -> dict:
    grid = sorted(grid or [round(1 - i * 0.01, 2) for i in range(0, 60)], reverse=True)
    chosen, trail = None, []
    for tau in grid:  # strictest (highest tau keeps most blurred) -> loosest
        k = sum(1 for c in uncaught_conf if c >= tau)
        ub = clopper_pearson_upper(k, n_secrets, delta)
        trail.append({"tau": tau, "misses": k, "upper_bound": ub})
        if ub > alpha:
            break
        chosen = tau
    return {"alpha": alpha, "delta": delta, "n_secrets": n_secrets, "tau": chosen, "trail": trail}
