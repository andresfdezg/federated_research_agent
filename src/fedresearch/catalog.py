"""Shared clinical vocabulary for the synthetic sites.

The node schema is a simplified, OMOP CDM-inspired layout that uses standard source
vocabularies (SNOMED CT, LOINC, RxNorm). The API refers to concepts by short keys
(e.g. ``type_2_diabetes``) and each node resolves them to codes locally.
"""

from dataclasses import dataclass
from datetime import date

DATA_FREEZE_DATE = date(2025, 12, 31)


@dataclass(frozen=True)
class Concept:
    code: str
    vocabulary: str
    name: str
    unit: str | None = None


@dataclass(frozen=True)
class Variant:
    gene: str
    variant_id: str
    hgvs: str | None
    clinical_significance: str
    description: str


CONDITIONS = {
    "type_2_diabetes": Concept("44054006", "SNOMED", "Type 2 diabetes mellitus"),
    "essential_hypertension": Concept("59621000", "SNOMED", "Essential hypertension"),
    "coronary_artery_disease": Concept("53741008", "SNOMED", "Coronary arteriosclerosis"),
    "breast_cancer": Concept("254837009", "SNOMED", "Malignant neoplasm of breast"),
}

MEASUREMENTS = {
    "bmi": Concept("39156-5", "LOINC", "Body mass index", "kg/m2"),
    "ldl_cholesterol": Concept(
        "13457-7", "LOINC", "Cholesterol in LDL [Mass/volume] in Serum or Plasma by calculation", "mg/dL"
    ),
    "hba1c": Concept("4548-4", "LOINC", "Hemoglobin A1c/Hemoglobin.total in Blood", "%"),
}

DRUGS = {
    "atorvastatin": Concept("83367", "RxNorm", "atorvastatin"),
    "metformin": Concept("6809", "RxNorm", "metformin"),
}

VARIANTS = {
    "BRCA1": Variant("BRCA1", "rs80357914", "NM_007294.4:c.68_69del", "pathogenic", "185delAG founder variant"),
    "BRCA2": Variant("BRCA2", "rs80359550", "NM_000059.4:c.5946del", "pathogenic", "6174delT founder variant"),
    "LDLR": Variant(
        "LDLR", "LDLR_LOF", None, "pathogenic", "loss-of-function variant (familial hypercholesterolaemia)"
    ),
    "TCF7L2": Variant("TCF7L2", "rs7903146", None, "risk_factor", "T allele, type 2 diabetes risk"),
    "APOE": Variant("APOE", "rs429358", None, "risk_factor", "C allele (tags APOE e4)"),
}


def catalog_summary() -> dict:
    """Machine-readable catalog exposed to clients (and to the agent) via node metadata."""
    return {
        "conditions": {k: c.name for k, c in CONDITIONS.items()},
        "measurements": {k: f"{c.name} ({c.unit})" for k, c in MEASUREMENTS.items()},
        "drugs": {k: c.name for k, c in DRUGS.items()},
        "genes": {k: f"{v.variant_id} — {v.description}" for k, v in VARIANTS.items()},
    }
