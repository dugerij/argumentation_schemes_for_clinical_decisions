from dataclasses import dataclass


@dataclass(frozen=True)
class SourceVocabulary:
    name: str
    category: str
    description: str


DIAGNOSIS_VOCABS = (
    SourceVocabulary("ICD10CM", "diagnosis", "ICD-10-CM diagnosis codes"),
    SourceVocabulary(
        "SNOMEDCT_US",
        "diagnosis",
        "SNOMED CT clinical findings and disorders",
    ),
)

LAB_VOCABS = (
    SourceVocabulary(
        "LNC",
        "lab_or_measurement",
        "LOINC laboratory and clinical observations",
    ),
)

GENERAL_CLINICAL_VOCABS = (
    SourceVocabulary(
        "SNOMEDCT_US",
        "clinical_finding",
        "SNOMED CT broad clinical terminology",
    ),
    SourceVocabulary("MSH", "biomedical_topic", "MeSH biomedical descriptors"),
)

DEFAULT_SOURCE_VOCABS = (
    *DIAGNOSIS_VOCABS,
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
    "Diagnostic Procedure": "therapy_or_procedure",
    "Laboratory Procedure": "lab_or_measurement",
    "Laboratory or Test Result": "lab_or_measurement",
    "Body Part, Organ, or Organ Component": "anatomy",
}


def category_for(
    source_vocabulary: str | None,
    semantic_type: str | None = None,
) -> str | None:
    if semantic_type and semantic_type in SEMANTIC_TYPE_CATEGORY:
        return SEMANTIC_TYPE_CATEGORY[semantic_type]
    if source_vocabulary and source_vocabulary in SOURCE_CATEGORY:
        return SOURCE_CATEGORY[source_vocabulary]
    return None
