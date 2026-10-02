import json
import math

import pandas as pd
import pytest
import statsmodels.api as sm
from fastapi.testclient import TestClient
from pydantic import ValidationError

from fedresearch.data.generate import MODEL
from fedresearch.federation.meta import SiteEstimate, fixed_effect
from fedresearch.node.app import create_app
from fedresearch.node.privacy import AuditLog
from fedresearch.node.spec import CohortSpec, CountRequest, RegressionRequest, SummaryRequest

T2D_TCF7L2 = RegressionRequest.model_validate({
    "outcome": {"type": "has_condition", "condition": "type_2_diabetes"},
    "covariates": [
        {"type": "allele_count", "gene": "TCF7L2"},
        {"type": "age"},
        {"type": "measurement", "measurement": "bmi"},
    ],
    "family": "logistic",
})


def test_count_everyone_is_rounded_per_policy(engines):
    res = engines["cardio_cohort"].count(CountRequest())
    assert res["status"] == "ok"
    assert res["result"]["count"] == 6000  # rounded to 10


def test_small_cells_are_suppressed(engines):
    # Male BRCA1 carriers with breast cancer: the data contains none.
    cohort = CohortSpec.model_validate({"include": [
        {"type": "sex", "sex": "M"},
        {"type": "variant", "gene": "BRCA1"},
        {"type": "condition", "condition": "breast_cancer"},
    ]})
    for engine in engines.values():
        assert engine.count(CountRequest(cohort=cohort))["status"] == "suppressed"


def test_unknown_vocabulary_is_rejected_with_allowed_values():
    with pytest.raises(ValidationError) as exc:
        CohortSpec.model_validate({"include": [{"type": "condition", "condition": "diabetes"}]})
    assert "type_2_diabetes" in str(exc.value)


def test_extra_fields_are_rejected():
    with pytest.raises(ValidationError):
        CountRequest.model_validate({"cohort": {}, "sql": "SELECT * FROM person"})


def test_brca_carriers_are_diagnosed_younger(engines):
    def mean_age(**cohort):
        req = SummaryRequest(
            cohort=CohortSpec.model_validate(cohort),
            variable={"type": "age_at_diagnosis", "condition": "breast_cancer"},
        )
        return engines["hospital_central"].summary(req)

    brca = [{"type": "variant", "gene": "BRCA1"}, {"type": "variant", "gene": "BRCA2"}]
    carriers = mean_age(include=brca[:1])
    non_carriers = mean_age(exclude=brca)
    assert carriers["status"] == non_carriers["status"] == "ok"
    assert carriers["result"]["mean"] < non_carriers["result"]["mean"] - 8


def test_federated_estimate_matches_centralised_and_truth(engines):
    per_site = []
    for name, engine in engines.items():
        res = engine.regression(T2D_TCF7L2)
        assert res["status"] == "ok", res
        coef = next(c for c in res["result"]["coefficients"] if c["term"] == "alleles:TCF7L2")
        per_site.append(SiteEstimate(name, coef["estimate"], coef["std_error"]))
    federated = fixed_effect(per_site)

    # Centralised reference: pool row-level data (only possible in this test harness).
    variables = [T2D_TCF7L2.outcome, *T2D_TCF7L2.covariates]
    pooled = pd.concat([e.frame(T2D_TCF7L2.cohort, variables) for e in engines.values()]).dropna()
    fit = sm.Logit(pooled["x0"], sm.add_constant(pooled[["x1", "x2", "x3"]])).fit(disp=0)

    assert federated["estimate"] == pytest.approx(fit.params["x1"], abs=0.03)
    truth = MODEL["type_2_diabetes"]["tcf7l2_per_allele"]
    assert federated["ci_low"] < truth < federated["ci_high"]
    assert math.exp(federated["estimate"]) == pytest.approx(1.4, abs=0.15)


def test_api_audits_released_and_rejected_requests(engines, tmp_path):
    audit = AuditLog(tmp_path / "audit.jsonl")
    client = TestClient(create_app(engines["hospital_central"], audit))

    ok = client.post("/v1/count", json={"cohort": {}}, headers={"X-Requester": "agent-test"})
    assert ok.status_code == 200 and ok.json()["status"] == "ok"
    bad = client.post("/v1/count", json={"cohort": {}, "sql": "DROP TABLE person"})
    assert bad.status_code == 422 and bad.json()["status"] == "rejected"

    entries = [json.loads(line) for line in audit.path.read_text().splitlines()]
    assert [e["status"] for e in entries] == ["ok", "rejected"]
    assert entries[0]["requester"] == "agent-test"
    assert entries[0]["released"] == {"count": 4000}
