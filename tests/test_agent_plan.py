import math

import pytest

from fedresearch.agent.plan import AnalysisPlan, AnalysisStep, render_result, validate_plan
from fedresearch.federation.combine import combine_count, combine_regression, combine_summary, site_statuses


def step(**kwargs) -> AnalysisStep:
    return AnalysisStep.model_validate({"purpose": "test", **kwargs})


def test_valid_group_comparison_passes():
    s = step(
        analysis="regression",
        cohort={"include": [{"type": "condition", "condition": "breast_cancer"}]},
        outcome={"type": "age_at_diagnosis", "condition": "breast_cancer"},
        covariates=[{"type": "variant_carrier", "gene": "BRCA1"}],
        family="linear",
    )
    assert validate_plan(AnalysisPlan(rationale="r", steps=[s])) == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (  # outcome fixed by the cohort definition
            dict(
                analysis="regression",
                cohort={"include": [{"type": "condition", "condition": "type_2_diabetes"}]},
                outcome={"type": "has_condition", "condition": "type_2_diabetes"},
                covariates=[{"type": "age"}],
                family="logistic",
            ),
            "outcome",
        ),
        (  # group indicator fixed by the cohort definition
            dict(
                analysis="regression",
                cohort={"include": [{"type": "variant", "gene": "BRCA1"}]},
                outcome={"type": "has_condition", "condition": "breast_cancer"},
                covariates=[{"type": "variant_carrier", "gene": "BRCA1"}],
                family="logistic",
            ),
            "constant",
        ),
        (  # contradictory cohort
            dict(
                analysis="count",
                cohort={"include": [{"type": "sex", "sex": "F"}], "exclude": [{"type": "sex", "sex": "F"}]},
            ),
            "always empty",
        ),
        (dict(analysis="summary"), "needs 'variable'"),
    ],
)
def test_invalid_steps_are_explained(kwargs, message):
    errors = validate_plan(AnalysisPlan(rationale="r", steps=[step(**kwargs)]))
    assert errors and message in errors[0]


def _regression_response(estimate: float, se: float) -> dict:
    return {
        "status": "ok",
        "result": {
            "family": "logistic", "outcome": "has:type_2_diabetes", "n": 1000, "n_events": 100, "converged": True,
            "coefficients": [
                {"term": "const", "estimate": -2.0, "std_error": 0.1},
                {"term": "carrier:TCF7L2", "estimate": estimate, "std_error": se},
            ],
        },
    }


def test_combine_regression_skips_suppressed_nodes_and_reports_odds_ratios():
    responses = {
        "a": _regression_response(0.30, 0.05),
        "b": _regression_response(0.40, 0.10),
        "c": {"status": "suppressed", "reason": "fewer than 10 events or non-events"},
    }
    combined = combine_regression(responses)
    term = combined["terms"]["carrier:TCF7L2"]
    assert [s["site"] for s in term["per_site"]] == ["a", "b"]
    assert combined["n_total"] == 2000
    assert term["fixed"]["odds_ratio"] == pytest.approx(math.exp(term["fixed"]["estimate"]))

    s = step(
        analysis="regression", outcome={"type": "has_condition", "condition": "type_2_diabetes"},
        covariates=[{"type": "variant_carrier", "gene": "TCF7L2"}], family="logistic",
    )
    text = render_result(s, site_statuses(responses), combined)
    assert "c suppressed: fewer than 10" in text
    assert "OR 1.37" in text  # exp(0.32), the precision-weighted estimate


def test_combine_count_and_summary():
    counts = {"a": {"status": "ok", "result": {"count": 10}}, "b": {"status": "suppressed", "reason": "small"}}
    assert combine_count(counts) == {"total": 10, "per_site": {"a": 10}}

    props = {
        "a": {"status": "ok", "result": {"variable": "on:metformin", "n": 100, "n_positive": 80, "proportion": 0.8}},
        "b": {"status": "ok", "result": {"variable": "on:metformin", "n": 300, "n_positive": 270, "proportion": 0.9}},
    }
    assert combine_summary(props)["pooled"]["proportion"] == pytest.approx(350 / 400)
