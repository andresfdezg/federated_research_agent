import numpy as np
import pytest

from fedresearch.federation.meta import SiteEstimate, fixed_effect, pool_summaries, random_effects


def test_pool_summaries_matches_pooled_data():
    rng = np.random.default_rng(0)
    groups = [rng.normal(mu, sd, n) for mu, sd, n in [(5, 1, 50), (7, 2, 120), (6, 0.5, 30)]]
    pooled = np.concatenate(groups)
    n, mean, sd = pool_summaries([(len(g), g.mean(), g.std(ddof=1)) for g in groups])
    assert n == len(pooled)
    assert mean == pytest.approx(pooled.mean())
    assert sd == pytest.approx(pooled.std(ddof=1))


def test_fixed_effect_weights_by_precision():
    result = fixed_effect([SiteEstimate("a", 1.0, 0.1), SiteEstimate("b", 2.0, 0.2)])
    assert result["estimate"] == pytest.approx((1 / 0.01 * 1 + 1 / 0.04 * 2) / (1 / 0.01 + 1 / 0.04))
    assert result["ci_low"] < result["estimate"] < result["ci_high"]


def test_random_effects_reduces_to_fixed_when_homogeneous():
    estimates = [SiteEstimate("a", 0.5, 0.1), SiteEstimate("b", 0.5, 0.2), SiteEstimate("c", 0.5, 0.15)]
    re, fe = random_effects(estimates), fixed_effect(estimates)
    assert re["tau2"] == 0
    assert re["estimate"] == pytest.approx(fe["estimate"])


def test_random_effects_detects_heterogeneity():
    re = random_effects([SiteEstimate("a", 0.0, 0.05), SiteEstimate("b", 1.0, 0.05), SiteEstimate("c", 2.0, 0.05)])
    assert re["tau2"] > 0
    assert re["i2"] > 0.9
