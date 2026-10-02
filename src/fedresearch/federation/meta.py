"""Combine per-node aggregates into federated results without touching row-level data."""

import math
from dataclasses import dataclass

from scipy import stats

Z_975 = stats.norm.ppf(0.975)


@dataclass(frozen=True)
class SiteEstimate:
    site: str
    estimate: float
    std_error: float


def pool_summaries(summaries: list[tuple[int, float, float]]) -> tuple[int, float, float]:
    """Exact pooled (n, mean, sample SD) from per-site (n, mean, sample SD)."""
    n = sum(s[0] for s in summaries)
    mean = sum(ni * mi for ni, mi, _ in summaries) / n
    ss = sum((ni - 1) * sdi**2 + ni * (mi - mean) ** 2 for ni, mi, sdi in summaries)
    return n, mean, math.sqrt(ss / (n - 1))


def _inference(estimate: float, std_error: float) -> dict:
    z = estimate / std_error
    return {
        "estimate": estimate,
        "std_error": std_error,
        "ci_low": estimate - Z_975 * std_error,
        "ci_high": estimate + Z_975 * std_error,
        "z": z,
        "p_value": float(2 * stats.norm.sf(abs(z))),
    }


def fixed_effect(estimates: list[SiteEstimate]) -> dict:
    """Inverse-variance weighted fixed-effect meta-analysis."""
    weights = [1 / e.std_error**2 for e in estimates]
    total = sum(weights)
    pooled = sum(w * e.estimate for w, e in zip(weights, estimates)) / total
    return {"method": "fixed_effect", "k": len(estimates), **_inference(pooled, math.sqrt(1 / total))}


def random_effects(estimates: list[SiteEstimate]) -> dict:
    """DerSimonian–Laird random-effects meta-analysis with heterogeneity statistics."""
    weights = [1 / e.std_error**2 for e in estimates]
    total = sum(weights)
    fe = sum(w * e.estimate for w, e in zip(weights, estimates)) / total
    q = sum(w * (e.estimate - fe) ** 2 for w, e in zip(weights, estimates))
    df = len(estimates) - 1
    c = total - sum(w**2 for w in weights) / total
    tau2 = max(0.0, (q - df) / c) if df > 0 else 0.0
    re_weights = [1 / (e.std_error**2 + tau2) for e in estimates]
    re_total = sum(re_weights)
    pooled = sum(w * e.estimate for w, e in zip(re_weights, estimates)) / re_total
    return {
        "method": "random_effects_dl",
        "k": len(estimates),
        **_inference(pooled, math.sqrt(1 / re_total)),
        "tau2": tau2,
        "q": q,
        "q_p_value": float(stats.chi2.sf(q, df)) if df > 0 else None,
        "i2": max(0.0, (q - df) / q) if q > 0 else 0.0,
    }
