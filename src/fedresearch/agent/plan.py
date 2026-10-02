"""The analysis plan the agent proposes, its validation, and how plans and results are shown to humans and the LLM."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from fedresearch.node.spec import (
    AgeCriterion,
    CohortSpec,
    ConditionCriterion,
    CountRequest,
    DrugCriterion,
    MeasurementCriterion,
    RegressionRequest,
    SexCriterion,
    SummaryRequest,
    Variable,
    VariantCriterion,
)

Request = CountRequest | SummaryRequest | RegressionRequest


class AnalysisStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    purpose: str = Field(description="One sentence: what this step tells us")
    analysis: Literal["count", "summary", "regression"]
    cohort: CohortSpec = Field(default_factory=CohortSpec)
    variable: Variable | None = Field(None, description="summary only")
    outcome: Variable | None = Field(None, description="regression only")
    covariates: list[Variable] = Field(
        default_factory=list, description="regression only: exposure of interest first, then adjustment variables"
    )
    family: Literal["linear", "logistic"] | None = Field(None, description="regression only")

    def to_request(self) -> Request:
        if self.analysis == "count":
            return CountRequest(cohort=self.cohort)
        if self.analysis == "summary":
            if self.variable is None:
                raise ValueError("a summary step needs 'variable'")
            return SummaryRequest(cohort=self.cohort, variable=self.variable)
        if self.outcome is None or self.family is None or not self.covariates:
            raise ValueError("a regression step needs 'outcome', 'covariates' and 'family'")
        return RegressionRequest(
            cohort=self.cohort, outcome=self.outcome, covariates=self.covariates, family=self.family
        )


class AnalysisPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rationale: str = Field(description="How the steps together answer the question, including confounders considered")
    steps: list[AnalysisStep] = Field(min_length=1, max_length=5)


def _fixed_term(criterion) -> str | None:
    """Variable term that a cohort criterion makes constant (e.g. include breast_cancer -> has:breast_cancer)."""
    match criterion:
        case ConditionCriterion():
            return f"has:{criterion.condition}"
        case VariantCriterion():
            return f"carrier:{criterion.gene}"
        case DrugCriterion():
            return f"on:{criterion.drug}"
        case SexCriterion():
            return "male"
    return None


def validate_step(step: AnalysisStep) -> list[str]:
    try:
        step.to_request()
    except (ValueError, ValidationError) as exc:
        return [str(exc)]
    errors = []
    overlap = set(map(repr, step.cohort.include)) & set(map(repr, step.cohort.exclude))
    if overlap:
        errors.append("the cohort includes and excludes the same criterion, so it is always empty")
    fixed = {t for c in [*step.cohort.include, *step.cohort.exclude] if (t := _fixed_term(c))}
    if step.outcome is not None and step.outcome.term() in fixed:
        errors.append(
            f"the outcome '{step.outcome.term()}' is constant in this cohort (the cohort definition fixes it); "
            "use a cohort containing people both with and without the outcome, usually everyone"
        )
    for v in [step.variable, *step.covariates]:
        if v is not None and v.term() in fixed:
            errors.append(
                f"'{v.term()}' is fixed by the cohort definition, so it is constant; "
                "to compare groups, keep both groups in the cohort and use the variable as a covariate"
            )
    return errors


def validate_plan(plan: AnalysisPlan) -> list[str]:
    return [f"step {i}: {e}" for i, step in enumerate(plan.steps, 1) for e in validate_step(step)]


# --- rendering ------------------------------------------------------------------------------


def describe_criterion(c) -> str:
    match c:
        case ConditionCriterion():
            return f"has {c.condition}"
        case VariantCriterion():
            return f"{c.gene} carrier" + (" (homozygous)" if c.min_alt_alleles == 2 else "")
        case DrugCriterion():
            return f"on {c.drug}"
        case SexCriterion():
            return "female" if c.sex == "F" else "male"
        case AgeCriterion() | MeasurementCriterion():
            name = "age" if isinstance(c, AgeCriterion) else c.measurement
            if c.min is not None and c.max is not None:
                return f"{name} {c.min:g}–{c.max:g}"
            return f"{name} ≥ {c.min:g}" if c.min is not None else f"{name} ≤ {c.max:g}"
    return repr(c)


def describe_cohort(cohort: CohortSpec) -> str:
    parts = [" AND ".join(map(describe_criterion, cohort.include)) or "everyone"]
    if cohort.exclude:
        parts.append("excluding " + ", ".join(map(describe_criterion, cohort.exclude)))
    return " ".join(parts)


def describe_step(step: AnalysisStep) -> str:
    cohort = describe_cohort(step.cohort)
    if step.analysis == "count":
        return f"Count people: {cohort}"
    if step.analysis == "summary":
        return f"Summarise {step.variable.term()} in: {cohort}"
    covariates = " + ".join(v.term() for v in step.covariates)
    return f"{step.family.capitalize()} regression {step.outcome.term()} ~ {covariates} in: {cohort}"


def _p(p: float) -> str:
    return f"{p:.1e}" if p < 0.001 else f"{p:.3f}"


def render_result(step: AnalysisStep, sites: dict[str, str], combined: dict) -> str:
    """Compact, fully computed text the LLM can quote but does not need to do arithmetic on."""
    lines = [describe_step(step), "  sites: " + "; ".join(f"{k} {v}" for k, v in sites.items())]
    if step.analysis == "count":
        lines.append(f"  total (available sites): {combined['total']}  per site: {combined['per_site']}")
    elif step.analysis == "summary":
        pooled = combined["pooled"]
        if pooled is None:
            lines.append("  no site released a result")
        elif combined["kind"] == "proportion":
            lines.append(f"  pooled: {pooled['n_positive']}/{pooled['n']} = {pooled['proportion']:.1%}")
            lines += [f"    {s}: n={r['n']}, {r['proportion']:.1%}" for s, r in combined["per_site"].items()]
        else:
            lines.append(f"  pooled: n={pooled['n']}, mean={pooled['mean']:.2f}, sd={pooled['sd']:.2f}")
            lines += [f"    {s}: n={r['n']}, mean={r['mean']:.2f}, sd={r['sd']:.2f}" for s, r in combined["per_site"].items()]
    else:
        if not combined["terms"]:
            lines.append("  no site released a result")
        else:
            logistic = combined["family"] == "logistic"
            events = f", events={combined['n_events_total']}" if logistic else ""
            lines.append(f"  pooled n={combined['n_total']}{events}")
            for term, t in combined["terms"].items():
                for label, r in (("fixed-effect", t["fixed"]), ("random-effects", t["random"])):
                    if r is None:
                        continue
                    if logistic:
                        est = f"OR {r['odds_ratio']:.3f} (95% CI {r['or_ci_low']:.3f}–{r['or_ci_high']:.3f})"
                    else:
                        est = f"beta {r['estimate']:.3f} (95% CI {r['ci_low']:.3f}–{r['ci_high']:.3f})"
                    extra = f", I²={r['i2']:.0%}" if label == "random-effects" else ""
                    lines.append(f"  {term}: {label} {est}, p={_p(r['p_value'])}{extra} [{r['k']} sites]")
                per_site = ", ".join(
                    f"{s['site']} " + (f"OR {s['odds_ratio']:.3f}" if logistic else f"beta {s['estimate']:.3f}")
                    for s in t["per_site"]
                )
                lines.append(f"    per site: {per_site}")
    return "\n".join(lines)
