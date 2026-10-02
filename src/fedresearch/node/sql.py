"""Compile typed specs to parameterised DuckDB SQL. User input only ever reaches SQL as bound parameters."""

from fedresearch.catalog import CONDITIONS, DATA_FREEZE_DATE, DRUGS, MEASUREMENTS
from fedresearch.node.spec import (
    AgeAtDiagnosisVar,
    AgeCriterion,
    AgeVar,
    AlleleCountVar,
    CarrierVar,
    CohortSpec,
    ConditionCriterion,
    DrugCriterion,
    HasConditionVar,
    MaleVar,
    MeasurementCriterion,
    MeasurementVar,
    OnDrugVar,
    SexCriterion,
    VariantCriterion,
)

AGE_SQL = f"date_diff('day', birth_date, DATE '{DATA_FREEZE_DATE.isoformat()}') / 365.25"
LATEST_MEASUREMENT_SQL = (
    "SELECT person_id, arg_max(value_as_number, measurement_date) AS value "
    "FROM measurement WHERE concept_code = ? GROUP BY person_id"
)
_UNBOUNDED = 1e12


def criterion_sql(c) -> tuple[str, list]:
    """SQL selecting the person_ids that satisfy one criterion."""
    match c:
        case ConditionCriterion():
            return "SELECT person_id FROM condition_occurrence WHERE concept_code = ?", [CONDITIONS[c.condition].code]
        case VariantCriterion():
            return (
                "SELECT person_id FROM genomic_variant WHERE gene = ? AND alt_allele_count >= ?",
                [c.gene, c.min_alt_alleles],
            )
        case DrugCriterion():
            return "SELECT person_id FROM drug_exposure WHERE concept_code = ?", [DRUGS[c.drug].code]
        case SexCriterion():
            return "SELECT person_id FROM person WHERE sex = ?", [c.sex]
        case AgeCriterion():
            return (
                f"SELECT person_id FROM person WHERE {AGE_SQL} BETWEEN ? AND ?",
                [c.min if c.min is not None else 0, c.max if c.max is not None else 200],
            )
        case MeasurementCriterion():
            return (
                f"SELECT person_id FROM ({LATEST_MEASUREMENT_SQL}) WHERE value BETWEEN ? AND ?",
                [
                    MEASUREMENTS[c.measurement].code,
                    c.min if c.min is not None else -_UNBOUNDED,
                    c.max if c.max is not None else _UNBOUNDED,
                ],
            )
    raise TypeError(f"unsupported criterion: {c!r}")


def cohort_sql(cohort: CohortSpec) -> tuple[str, list]:
    clauses, params = [], []
    for negate, criteria in ((False, cohort.include), (True, cohort.exclude)):
        for c in criteria:
            sql, p = criterion_sql(c)
            clauses.append(f"person_id {'NOT IN' if negate else 'IN'} ({sql})")
            params += p
    return f"SELECT person_id FROM person WHERE {' AND '.join(clauses) or 'TRUE'}", params


def variable_sql(v) -> tuple[str, list, int | None]:
    """SQL yielding (person_id, value) for a variable, plus the fill value for people without a row."""
    match v:
        case AgeVar():
            return f"SELECT person_id, {AGE_SQL} AS value FROM person", [], None
        case MaleVar():
            return "SELECT person_id, CASE WHEN sex = 'M' THEN 1 ELSE 0 END AS value FROM person", [], None
        case MeasurementVar():
            return LATEST_MEASUREMENT_SQL, [MEASUREMENTS[v.measurement].code], None
        case AgeAtDiagnosisVar():
            return (
                "SELECT o.person_id, date_diff('day', p.birth_date, min(o.condition_start_date)) / 365.25 AS value "
                "FROM condition_occurrence o JOIN person p ON p.person_id = o.person_id "
                "WHERE o.concept_code = ? GROUP BY o.person_id, p.birth_date",
                [CONDITIONS[v.condition].code],
                None,
            )
        case HasConditionVar():
            return (
                "SELECT DISTINCT person_id, 1 AS value FROM condition_occurrence WHERE concept_code = ?",
                [CONDITIONS[v.condition].code],
                0,
            )
        case CarrierVar():
            return "SELECT DISTINCT person_id, 1 AS value FROM genomic_variant WHERE gene = ?", [v.gene], 0
        case AlleleCountVar():
            return (
                "SELECT person_id, max(alt_allele_count) AS value FROM genomic_variant WHERE gene = ? GROUP BY person_id",
                [v.gene],
                0,
            )
        case OnDrugVar():
            return (
                "SELECT DISTINCT person_id, 1 AS value FROM drug_exposure WHERE concept_code = ?",
                [DRUGS[v.drug].code],
                0,
            )
    raise TypeError(f"unsupported variable: {v!r}")


def frame_sql(cohort: CohortSpec, variables: list) -> tuple[str, list]:
    """One row per cohort member with columns x0..xN for the requested variables."""
    sql, params = cohort_sql(cohort)
    selects, joins = ["c.person_id"], []
    for i, v in enumerate(variables):
        vsql, vparams, fill = variable_sql(v)
        selects.append(f"{f'v{i}.value' if fill is None else f'COALESCE(v{i}.value, {fill})'} AS x{i}")
        joins.append(f"LEFT JOIN ({vsql}) v{i} ON v{i}.person_id = c.person_id")
        params += vparams
    return f"SELECT {', '.join(selects)} FROM ({sql}) c {' '.join(joins)}", params
