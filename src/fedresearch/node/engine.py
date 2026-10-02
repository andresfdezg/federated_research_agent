"""Runs analyses against a node's local database and applies the privacy policy before anything is released."""

from dataclasses import asdict
from pathlib import Path

import duckdb
import pandas as pd
import statsmodels.api as sm

from fedresearch.catalog import DATA_FREEZE_DATE, catalog_summary
from fedresearch.node.privacy import PrivacyPolicy
from fedresearch.node.spec import CohortSpec, CountRequest, RegressionRequest, SummaryRequest
from fedresearch.node.sql import cohort_sql, frame_sql

VARIABLE_TYPES = {
    "age": "age in years at data freeze",
    "male": "1 if male else 0",
    "measurement": "most recent value of a measurement",
    "age_at_diagnosis": "age at first record of a condition",
    "has_condition": "1 if the person has the condition",
    "variant_carrier": "1 if the person carries the gene's variant",
    "allele_count": "number of alternate alleles (0/1/2)",
    "on_drug": "1 if the person was ever prescribed the drug",
}


class NodeEngine:
    def __init__(self, name: str, db_path: Path, policy: PrivacyPolicy, description: str = ""):
        self.name = name
        self.description = description
        self.policy = policy
        self._con = duckdb.connect(str(db_path), read_only=True)

    # Row-level access stays inside the node: only the public methods below release results.
    def _scalar(self, sql: str, params: list) -> int:
        return self._con.cursor().execute(sql, params).fetchone()[0]

    def frame(self, cohort: CohortSpec, variables: list) -> pd.DataFrame:
        sql, params = frame_sql(cohort, variables)
        return self._con.cursor().execute(sql, params).df().drop(columns="person_id").astype(float)

    def _ok(self, result: dict) -> dict:
        return {"node": self.name, "status": "ok", "result": result}

    def _suppressed(self, reason: str) -> dict:
        return {"node": self.name, "status": "suppressed", "reason": reason}

    def _error(self, reason: str) -> dict:
        return {"node": self.name, "status": "error", "reason": reason}

    def metadata(self) -> dict:
        n = self._scalar("SELECT count(*) FROM person", [])
        return {
            "node": self.name,
            "description": self.description,
            "approximate_size": int(round(n, -2)),
            "data_freeze_date": DATA_FREEZE_DATE.isoformat(),
            "privacy_policy": asdict(self.policy),
            "catalog": catalog_summary(),
            "variable_types": VARIABLE_TYPES,
            "analyses": ["count", "summary", "regression"],
        }

    def count(self, req: CountRequest) -> dict:
        sql, params = cohort_sql(req.cohort)
        n = self._scalar(f"SELECT count(*) FROM ({sql})", params)
        if not self.policy.releasable(n):
            return self._suppressed(f"cohort size below minimum cell count ({self.policy.min_cell_count})")
        return self._ok({"count": self.policy.round_count(n)})

    def summary(self, req: SummaryRequest) -> dict:
        x = self.frame(req.cohort, [req.variable])["x0"].dropna()
        n, k = len(x), self.policy.min_cell_count
        if req.variable.binary:
            positives = int(x.sum())
            if min(positives, n - positives) < k:
                return self._suppressed(f"a group (with/without '{req.variable.term()}') has fewer than {k} people")
            return self._ok({
                "variable": req.variable.term(),
                "n": self.policy.round_count(n),
                "n_positive": self.policy.round_count(positives),
                "proportion": round(positives / n, 3),
            })
        if not self.policy.releasable(n):
            return self._suppressed(f"fewer than {k} people with a value for '{req.variable.term()}'")
        return self._ok({
            "variable": req.variable.term(),
            "n": self.policy.round_count(n),
            "mean": round(float(x.mean()), 3),
            "sd": round(float(x.std(ddof=1)), 3),
        })

    def regression(self, req: RegressionRequest) -> dict:
        df = self.frame(req.cohort, [req.outcome, *req.covariates]).dropna()
        n, k = len(df), self.policy.min_cell_count
        n_params = len(req.covariates) + 1
        if n < max(k, 10 * n_params):
            return self._suppressed(f"complete-case n is below max({k}, 10 per model parameter)")

        y = df["x0"]
        X = df.drop(columns="x0")
        X.columns = [v.term() for v in req.covariates]
        for v in req.covariates:
            col = X[v.term()]
            if col.nunique() < 2:
                return self._error(f"covariate '{v.term()}' is constant in this cohort; remove it")
            if v.binary and min(int(col.sum()), n - int(col.sum())) < k:
                return self._suppressed(f"covariate '{v.term()}' has a level with fewer than {k} people")

        events = None
        if req.family == "logistic":
            events = int(y.sum())
            if min(events, n - events) < k:
                return self._suppressed(f"fewer than {k} events or non-events")

        X = sm.add_constant(X, has_constant="add")
        try:
            if req.family == "logistic":
                fit = sm.Logit(y, X).fit(disp=0, maxiter=200)
                converged = bool(fit.mle_retvals.get("converged", False))
            else:
                fit = sm.OLS(y, X).fit()
                converged = True
        except Exception as exc:  # singular matrix, perfect separation, ...
            return self._error(f"model fit failed: {exc}")

        return self._ok({
            "family": req.family,
            "outcome": req.outcome.term(),
            "n": self.policy.round_count(n),
            "n_events": self.policy.round_count(events) if events is not None else None,
            "converged": converged,
            "coefficients": [
                {"term": t, "estimate": round(float(fit.params[t]), 6), "std_error": round(float(fit.bse[t]), 6)}
                for t in X.columns
            ],
        })
