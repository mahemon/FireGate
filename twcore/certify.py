"""Finite-sample tools for tripwire certificates (pure Python, no torch).

clopper_pearson_upper  one-sided upper confidence bound on a firing probability (Proposition 3.4)
calib_index            order-statistic index for PAC calibration (Theorem 3.5)
calibrate              per-component thresholds and the joint threshold
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional


def _log_binom_pmf(k: int, n: int, p: float) -> float:
    if p <= 0.0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if k == n else -math.inf
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k * math.log(p) + (n - k) * math.log1p(-p))


def binom_cdf(k: int, n: int, p: float) -> float:
    """P[Bin(n, p) <= k]."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    logs = [_log_binom_pmf(i, n, p) for i in range(0, k + 1)]
    m = max(logs)
    if m == -math.inf:
        return 0.0
    return min(1.0, math.exp(m) * sum(math.exp(v - m) for v in logs))


def binom_sf_ge(k: int, n: int, p: float) -> float:
    """P[Bin(n, p) >= k]."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    logs = [_log_binom_pmf(i, n, p) for i in range(k, n + 1)]
    m = max(logs)
    if m == -math.inf:
        return 0.0
    return min(1.0, math.exp(m) * sum(math.exp(v - m) for v in logs))


def clopper_pearson_upper(x: int, n: int, delta: float = 0.05) -> float:
    """Smallest p with P[Bin(n, p) <= x] <= delta, i.e. the one-sided (1-delta) upper bound."""
    if n <= 0:
        return 1.0
    if x >= n:
        return 1.0
    lo, hi = x / n, 1.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if binom_cdf(x, n, mid) > delta:
            lo = mid
        else:
            hi = mid
    return hi


def calib_index(n: int, eps: float, delta: float) -> Optional[int]:
    """Smallest k in 1..n with P[Bin(n, 1-eps) >= k] <= delta (1-based), or None if none exists."""
    for k in range(1, n + 1):
        if binom_sf_ge(k, n, 1.0 - eps) <= delta:
            return k
    return None


def eps_for_index(n: int, k: int, delta: float) -> float:
    """Smallest eps such that P[Bin(n, 1-eps) >= k] <= delta (the guarantee that order statistic k gives)."""
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if binom_sf_ge(k, n, 1.0 - mid) <= delta:
            hi = mid
        else:
            lo = mid
    return hi


def calibrate(scores: Dict[str, List[float]], eps: float, delta: float) -> dict:
    """PAC calibration (Theorem 3.5) with a Bonferroni split of delta over components.

    scores[c] are trajectory scores S = max_t w(u_t) of the old model on calibration prompts of component c.
    The tripwire fires iff S > kappa. Returns kappa = max_c kappa_c and, per component, the order statistic
    used, the guarantee eps_cert it gives (= eps unless the component has too few samples, in which case the
    largest score is used and eps_cert is the guarantee of that choice) and the in-sample firing rate.
    """
    comps = {c: sorted(float(s) for s in v) for c, v in scores.items() if len(v) > 0}
    if not comps:
        raise ValueError("no calibration scores")
    d = delta / len(comps)
    out = {}
    kappa = -math.inf
    for c, s in comps.items():
        n = len(s)
        k = calib_index(n, eps, d)
        feasible = k is not None
        if not feasible:
            k = n
        kap_c = s[k - 1]
        out[c] = {"n": n, "k": k, "kappa_c": kap_c, "feasible": feasible,
                  "eps_cert": eps if feasible else eps_for_index(n, k, d)}
        kappa = max(kappa, kap_c)
    for c, s in comps.items():
        out[c]["emp_rate"] = sum(1 for v in s if v > kappa) / len(s)
        # the joint threshold is at least kappa_c, so each component's guarantee still holds
    return {"kappa": kappa, "eps": eps, "delta": delta, "delta_per_component": d, "components": out}
