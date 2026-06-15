from dataclasses import dataclass


@dataclass(frozen=True)
class SourceVocabulary:
    name: str
    category: str
    description: str


DIAGNOSIS_VOCABS = (
    SourceVocabulary("ICD10CM", "diagnosis", "ICD-10-CM diagnosis codes"),
    SourceVocabulary("SNOMEDCT_US", "diagnosis", "SNOMED CT clinical findings and disorders"),
)

MEDICATION_VOCABS = (
    SourceVocabulary("RXNORM", "medication", "RxNorm normalized medication concepts"),
    SourceVocabulary("ATC", "medication", "Anatomical Therapeutic Chemical drug classes"),
)

THERAPY_PROCEDURE_VOCABS = (
    SourceVocabulary("SNOMEDCT_US", "therapy_or_procedure", "SNOMED CT procedures and interventions"),
    SourceVocabulary("CPT", "therapy_or_procedure", "Current Procedural Terminology"),
    SourceVocabulary("HCPCS", "therapy_or_procedure", "Healthcare Common Procedure Coding System"),
    SourceVocabulary("MEDCIN", "therapy_or_procedure", "MEDCIN clinical terminology"),
)

LAB_VOCABS = (
    SourceVocabulary("LNC", "lab_or_measurement", "LOINC laboratory and clinical observations"),
    SourceVocabulary("SNOMEDCT_US", "lab_or_measurement", "SNOMED CT observables and test results"),
)

GENERAL_CLINICAL_VOCABS = (
    SourceVocabulary("MEDCIN", "clinical_finding", "MEDCIN clinical findings and symptoms"),
    SourceVocabulary("SNOMEDCT_US", "clinical_finding", "SNOMED CT broad clinical terminology"),
    SourceVocabulary("MSH", "biomedical_topic", "MeSH biomedical descriptors"),
)

DEFAULT_SOURCE_VOCABS = (
    *DIAGNOSIS_VOCABS,
    *MEDICATION_VOCABS,
    *THERAPY_PROCEDURE_VOCABS,
    *LAB_VOCABS,
    *GENERAL_CLINICAL_VOCABS,
)

SOURCE_PRIORITY = tuple(dict.fromkeys(vocab.name for vocab in DEFAULT_SOURCE_VOCABS))

SOURCE_CATEGORY = {}
for vocab in DEFAULT_SOURCE_VOCABS:
    SOURCE_CATEGORY.setdefault(vocab.name, vocab.category)

SEMANTIC_TYPE_CATEGORY = {
    "Disease or Syndrome": "diagnosis",
    "Sign or Symptom": "clinical_finding",
    "Finding": "clinical_finding",
    "Pharmacologic Substance": "medication",
    "Clinical Drug": "medication",
    "Therapeutic or Preventive Procedure": "therapy_or_procedure",
    "Diagnostic Procedure": "therapy_or_procedure",
    "Laboratory Procedure": "lab_or_measurement",
    "Laboratory or Test Result": "lab_or_measurement",
    "Body Part, Organ, or Organ Component": "anatomy",
}


def category_for(source_vocabulary: str | None, semantic_type: str | None = None) -> str | None:
    if semantic_type and semantic_type in SEMANTIC_TYPE_CATEGORY:
        return SEMANTIC_TYPE_CATEGORY[semantic_type]
    if source_vocabulary and source_vocabulary in SOURCE_CATEGORY:
        return SOURCE_CATEGORY[source_vocabulary]
    return None
