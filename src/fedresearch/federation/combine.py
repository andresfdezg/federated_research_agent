"""Turn per-node responses into federated results. All arithmetic happens here, never in the LLM."""

import math

from fedresearch.federation.meta import SiteEstimate, fixed_effect, pool_summaries, random_effects


def _ok(responses: dict[str, dict]) -> dict[str, dict]:
    return {name: r["result"] for name, r in responses.items() if r.get("status") == "ok"}


def site_statuses(responses: dict[str, dict]) -> dict[str, str]:
    return {
        name: r["status"] if r.get("status") == "ok" else f"{r.get('status')}: {r.get('reason', '')}".rstrip(": ")
        for name, r in responses.items()
    }


def combine_count(responses: dict[str, dict]) -> dict:
    ok = _ok(responses)
    per_site = {name: r["count"] for name, r in ok.items()}
    return {"total": sum(per_site.values()) if ok else None, "per_site": per_site}


def combine_summary(responses: dict[str, dict]) -> dict:
    ok = _ok(responses)
    if not ok:
        return {"kind": None, "pooled": None, "per_site": {}}
    if "proportion" in next(iter(ok.values())):
        n = sum(r["n"] for r in ok.values())
        positives = sum(r["n_positive"] for r in ok.values())
        return {
            "kind": "proportion",
            "pooled": {"n": n, "n_positive": positives, "proportion": positives / n},
            "per_site": {name: {"n": r["n"], "proportion": r["proportion"]} for name, r in ok.items()},
        }
    n, mean, sd = pool_summaries([(r["n"], r["mean"], r["sd"]) for r in ok.values()])
    return {
        "kind": "mean",
        "pooled": {"n": n, "mean": mean, "sd": sd},
        "per_site": {name: {"n": r["n"], "mean": r["mean"], "sd": r["sd"]} for name, r in ok.items()},
    }


def _with_odds_ratio(result: dict) -> dict:
    return result | {
        "odds_ratio": math.exp(result["estimate"]),
        "or_ci_low": math.exp(result["ci_low"]),
        "or_ci_high": math.exp(result["ci_high"]),
    }


def combine_regression(responses: dict[str, dict]) -> dict:
    ok = _ok(responses)
    if not ok:
        return {"family": None, "terms": {}, "n_total": None}
    first = next(iter(ok.values()))
    logistic = first["family"] == "logistic"
    terms = {}
    for term in (c["term"] for c in first["coefficients"] if c["term"] != "const"):
        estimates = [
            SiteEstimate(name, c["estimate"], c["std_error"])
            for name, r in ok.items()
            for c in r["coefficients"]
            if c["term"] == term
        ]
        fixed = fixed_effect(estimates)
        random = random_effects(estimates) if len(estimates) > 1 else None
        per_site = [
            {"site": e.site, "estimate": e.estimate, "std_error": e.std_error}
            | ({"odds_ratio": math.exp(e.estimate)} if logistic else {})
            for e in estimates
        ]
        if logistic:
            fixed = _with_odds_ratio(fixed)
            random = _with_odds_ratio(random) if random else None
        terms[term] = {"fixed": fixed, "random": random, "per_site": per_site}
    return {
        "family": first["family"],
        "outcome": first["outcome"],
        "n_total": sum(r["n"] for r in ok.values()),
        "n_events_total": sum(r["n_events"] for r in ok.values()) if logistic else None,
        "terms": terms,
    }
