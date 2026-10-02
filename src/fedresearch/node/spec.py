"""Typed query language accepted by a node.

Clients (including the agent) never send SQL to a node. They send a cohort
definition plus an analysis request, validated against a closed vocabulary and
compiled to parameterised SQL inside the node. Unknown fields are rejected, and
validation errors list the allowed values so an LLM can self-correct.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fedresearch.catalog import CONDITIONS, DRUGS, MEASUREMENTS, VARIANTS

ConditionKey = Literal[tuple(CONDITIONS)]
MeasurementKey = Literal[tuple(MEASUREMENTS)]
DrugKey = Literal[tuple(DRUGS)]
GeneKey = Literal[tuple(VARIANTS)]


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _check_range(spec: BaseModel) -> BaseModel:
    if spec.min is None and spec.max is None:
        raise ValueError("set at least one of 'min' or 'max'")
    if spec.min is not None and spec.max is not None and spec.min > spec.max:
        raise ValueError("'min' must not exceed 'max'")
    return spec


# --- cohort criteria -------------------------------------------------------------------------


class ConditionCriterion(_Spec):
    type: Literal["condition"]
    condition: ConditionKey


class VariantCriterion(_Spec):
    type: Literal["variant"]
    gene: GeneKey
    min_alt_alleles: int = Field(1, ge=1, le=2)


class DrugCriterion(_Spec):
    type: Literal["drug"]
    drug: DrugKey


class SexCriterion(_Spec):
    type: Literal["sex"]
    sex: Literal["F", "M"]


class AgeCriterion(_Spec):
    type: Literal["age"]
    min: float | None = Field(None, ge=0, le=120)
    max: float | None = Field(None, ge=0, le=120)

    @model_validator(mode="after")
    def check_range(self):
        return _check_range(self)


class MeasurementCriterion(_Spec):
    """Filter on each person's most recent value of a measurement."""

    type: Literal["measurement"]
    measurement: MeasurementKey
    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def check_range(self):
        return _check_range(self)


Criterion = Annotated[
    ConditionCriterion | VariantCriterion | DrugCriterion | SexCriterion | AgeCriterion | MeasurementCriterion,
    Field(discriminator="type"),
]


class CohortSpec(_Spec):
    """People matching ALL `include` criteria and NONE of the `exclude` criteria. Empty = everyone."""

    include: list[Criterion] = Field(default_factory=list, max_length=10)
    exclude: list[Criterion] = Field(default_factory=list, max_length=10)


# --- analysis variables ----------------------------------------------------------------------


class AgeVar(_Spec):
    binary: ClassVar[bool] = False
    type: Literal["age"]

    def term(self) -> str:
        return "age"


class MaleVar(_Spec):
    binary: ClassVar[bool] = True
    type: Literal["male"]

    def term(self) -> str:
        return "male"


class MeasurementVar(_Spec):
    """Most recent value of a measurement (missing if never measured)."""

    binary: ClassVar[bool] = False
    type: Literal["measurement"]
    measurement: MeasurementKey

    def term(self) -> str:
        return self.measurement


class AgeAtDiagnosisVar(_Spec):
    """Age at first record of a condition (missing for people without it)."""

    binary: ClassVar[bool] = False
    type: Literal["age_at_diagnosis"]
    condition: ConditionKey

    def term(self) -> str:
        return f"age_at_diagnosis:{self.condition}"


class HasConditionVar(_Spec):
    binary: ClassVar[bool] = True
    type: Literal["has_condition"]
    condition: ConditionKey

    def term(self) -> str:
        return f"has:{self.condition}"


class CarrierVar(_Spec):
    binary: ClassVar[bool] = True
    type: Literal["variant_carrier"]
    gene: GeneKey

    def term(self) -> str:
        return f"carrier:{self.gene}"


class AlleleCountVar(_Spec):
    """Number of alternate alleles (0, 1, 2): additive genetic model."""

    binary: ClassVar[bool] = False
    type: Literal["allele_count"]
    gene: GeneKey

    def term(self) -> str:
        return f"alleles:{self.gene}"


class OnDrugVar(_Spec):
    binary: ClassVar[bool] = True
    type: Literal["on_drug"]
    drug: DrugKey

    def term(self) -> str:
        return f"on:{self.drug}"


Variable = Annotated[
    AgeVar | MaleVar | MeasurementVar | AgeAtDiagnosisVar | HasConditionVar | CarrierVar | AlleleCountVar | OnDrugVar,
    Field(discriminator="type"),
]


# --- requests --------------------------------------------------------------------------------


class CountRequest(_Spec):
    cohort: CohortSpec = CohortSpec()


class SummaryRequest(_Spec):
    cohort: CohortSpec = CohortSpec()
    variable: Variable


class RegressionRequest(_Spec):
    cohort: CohortSpec = CohortSpec()
    outcome: Variable
    covariates: list[Variable] = Field(min_length=1, max_length=8)
    family: Literal["linear", "logistic"]

    @model_validator(mode="after")
    def _check(self) -> "RegressionRequest":
        if self.family == "logistic" and not self.outcome.binary:
            raise ValueError("logistic regression needs a binary outcome (male, has_condition, variant_carrier, on_drug)")
        terms = [self.outcome.term(), *(v.term() for v in self.covariates)]
        if len(set(terms)) != len(terms):
            raise ValueError("outcome and covariates must all be distinct variables")
        return self
