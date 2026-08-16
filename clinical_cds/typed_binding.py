"""Typed, executable patient-observation to clinical-criterion binding."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from clinical_cds.terminology.candidates import extract_candidate_terms


TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9/-]*")
CLAUSE_RE = re.compile(r"(?<=[.!?])\s+|[;\r\n]+")
PATIENT_CLAUSE_RE = re.compile(r"(?<=[.!?])\s+|[;:\r\n]+")
NEGATION_RE = re.compile(
    r"\b(?:no|not|without|denies|denied|negative|absent|absence of|"
    r"lacks|lack of)\b",
    re.IGNORECASE,
)
HISTORICAL_RE = re.compile(
    r"\b(?:history of|historical|previous|previously|prior|status post|s/p|"
    r"remote|years? ago)\b",
    re.IGNORECASE,
)
CURRENT_RE = re.compile(
    r"\b(?:current|currently|today|now|presented|acute|new|on admission)\b",
    re.IGNORECASE,
)
CURRENT_COMPARISON_RE = re.compile(
    r"\b(?:since|compared\s+(?:with|to))\s+(?:the\s+)?(?:previous|prior)\s+"
    r"(?:tracing|study|exam(?:ination)?|scan|test)\b",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(
    r"(?P<operator><=|>=|≤|≥|<|>|below|under|above|over|at least|at most|"
    r"more than|less than|greater than|exceeding|exceeds|exceeded|"
    r"surpassing|surpasses|surpassed)?\s*"
    r"(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>%|mmhg|mg/dl|mmol/l|ng/ml|pg/ml|µg/l|ug/l|mm|cm/sec|cm/s|"
    r"seconds?|minutes?|hours?|days?|weeks?|months?|years?)?",
    re.IGNORECASE,
)
RANGE_RE = re.compile(
    r"(?P<low>\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*"
    r"(?P<high>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>%|mmhg|mg/dl|mmol/l|ng/ml|pg/ml|µg/l|ug/l|mm|cm/sec|cm/s)?",
    re.IGNORECASE,
)
STOPWORDS = frozenset({
    "about", "after", "also", "and", "any", "are", "been", "before",
    "but", "can", "clinical", "could", "criterion", "criteria", "day",
    "days", "diagnosis", "diagnostic", "feature", "finding", "findings",
    "for", "from", "had", "has", "have", "his", "into", "more", "most",
    "level", "levels", "over", "patient", "present", "result", "results",
    "sign", "signs",
    "state", "symptom", "symptoms", "test", "tests", "than", "that", "the",
    "their", "there", "this", "was", "were", "who", "with",
})
POLARITY_ONLY_TOKENS = frozenset({
    "absent", "absence", "denied", "denies", "negative", "normal",
    "unremarkable", "without",
})
ABNORMAL_LAB_RE = re.compile(
    r"(?P<name>[A-Za-z][A-Za-z0-9/-]{1,30})\s*[-:=]\s*"
    r"(?P<operator>[<>]?)\s*(?P<value>\d+(?:\.\d+)?)\*",
    re.IGNORECASE,
)
ELEVATION_TOKENS = frozenset({
    "above", "elevated", "exceeded", "high", "increase", "increased",
})
NON_ST_ELEVATION_RE = re.compile(
    r"\b(?:non[-\s]?st[-\s]?elevation|no\s+st[-\s]?elevation)\b",
    re.IGNORECASE,
)
ST_DEPRESSION_RE = re.compile(
    r"(?:\bst(?:[-\s]+segment)?[-\s]+depression\b|"
    r"\b(?:horizontal|downsloping|upsloping|concordant)\s+std\b|"
    r"\bstd\s+(?:in\s+)?(?:lead\s+)?v\d\b)",
    re.IGNORECASE,
)
DIAGNOSTIC_TEST_RE = re.compile(
    r"\b(?:ct(?:pa)?|mri|mra|x[- ]?ray|radiograph|ultrasound|echo(?:cardiogram)?|"
    r"angiograph(?:y|ram)|endoscop(?:y|ic)|egd|colonoscopy|biopsy|pathology|"
    r"culture|pcr|scan|imaging|spirometr(?:y|ic)|pulmonary function test|pft|"
    r"stress test|monitoring|tracing)\b",
    re.IGNORECASE,
)
COMPLETED_RESULT_RE = re.compile(
    r"\b(?:show(?:ed|s)|demonstrat(?:ed|es)|reveal(?:ed|s)|confirm(?:ed|s)|"
    r"positive for|consistent with|diagnostic of|significant for|evidence of|"
    r"found|identified|visuali[sz](?:ed|es))\b",
    re.IGNORECASE,
)
NONFINAL_TEST_RE = re.compile(
    r"\b(?:ordered|planned|scheduled|pending|recommended|referred|consider(?:ed|ation)|"
    r"to (?:exclude|rule out|evaluate)|cannot exclude|possible|suspicious for|equivocal)\b",
    re.IGNORECASE,
)


class Polarity(str, Enum):
    PRESENT = "present"
    ABSENT = "absent"


class Temporality(str, Enum):
    CURRENT = "current"
    HISTORICAL = "historical"
    UNSPECIFIED = "unspecified"


@dataclass(frozen=True)
class Quantity:
    value: float
    unit: str
    operator: str = "="


@dataclass(frozen=True)
class ClinicalAtom:
    text: str
    tokens: tuple[str, ...]
    cuis: tuple[str, ...]
    polarity: Polarity
    temporality: Temporality
    quantities: tuple[Quantity, ...]


@dataclass(frozen=True)
class CriticalQuestionResult:
    question_id: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class TypedBindingAssessment:
    admissible: bool
    score: float
    matched_clause: str
    patient_atom: ClinicalAtom
    criterion_atom: ClinicalAtom
    critical_questions: tuple[CriticalQuestionResult, ...]


@dataclass(frozen=True)
class DiagnosticTestResultAssessment:
    admissible: bool
    score: float
    critical_questions: tuple[CriticalQuestionResult, ...]


def controlled_diagnosis_assertion(
    patient_finding: str,
    criterion_text: str,
    diagnosis_label: str,
    *,
    normalizer: object | None = None,
) -> bool:
    """Whether an admitted pair literally asserts a controlled diagnosis.

    This is deliberately narrower than clinical similarity.  The patient and
    warrant must denote the same proposition, and the warrant must itself
    denote the controlled graph diagnosis.  UMLS may standardise synonyms and
    abbreviations, but related concepts, symptoms, measurements, and
    complications do not acquire diagnosis authority.
    """
    patient = clinical_atom(patient_finding, normalizer)
    criterion = clinical_atom(criterion_text, normalizer)
    diagnosis = clinical_atom(diagnosis_label, normalizer)
    if (
        patient.polarity != Polarity.PRESENT
        or criterion.polarity != Polarity.PRESENT
        or patient.temporality == Temporality.HISTORICAL
    ):
        return False

    def equivalent(left: ClinicalAtom, right: ClinicalAtom) -> bool:
        left_cuis = set(left.cuis)
        right_cuis = set(right.cuis)
        if left_cuis and right_cuis and left_cuis & right_cuis:
            return True
        left_tokens = set(left.tokens) - POLARITY_ONLY_TOKENS
        right_tokens = set(right.tokens) - POLARITY_ONLY_TOKENS
        return bool(left_tokens and right_tokens) and (
            left_tokens <= right_tokens or right_tokens <= left_tokens
        )

    return equivalent(patient, criterion) and equivalent(criterion, diagnosis)


@dataclass(frozen=True)
class DeterministicBindingAuthority:
    """Whether a typed match is sufficiently executable to bypass semantic review."""

    authorized: bool
    authority_type: str
    detail: str


_MODALITY_PATTERNS = (
    ("ctpa", re.compile(r"\b(?:ctpa|ct pulmonary angiograph(?:y|ram))\b", re.I)),
    ("cta", re.compile(r"\b(?:cta|ct angiograph(?:y|ram))\b", re.I)),
    ("mra", re.compile(r"\b(?:mra|mr angiograph(?:y|ram))\b", re.I)),
    ("angiography", re.compile(r"\bangiograph(?:y|ram)\b", re.I)),
    ("endoscopy", re.compile(r"\b(?:endoscop(?:y|ic)|egd)\b", re.I)),
    ("ct", re.compile(r"\bct(?: scan| imaging)?\b", re.I)),
    ("mri", re.compile(r"\bmri?\b", re.I)),
    ("radiograph", re.compile(r"\b(?:x[- ]?ray|radiograph)\b", re.I)),
    ("ultrasound", re.compile(r"\b(?:ultrasound|sonograph(?:y|ic))\b", re.I)),
    ("echocardiography", re.compile(r"\b(?:echo|echocardiogram|echocardiography)\b", re.I)),
    ("spirometry", re.compile(r"\b(?:spirometr(?:y|ic)|pulmonary function test|pft)\b", re.I)),
    ("pathology", re.compile(r"\b(?:biopsy|pathology|histology)\b", re.I)),
    ("culture", re.compile(r"\bculture\b", re.I)),
    ("pcr", re.compile(r"\bpcr\b", re.I)),
    ("stress_test", re.compile(r"\b(?:stress test|ett)\b", re.I)),
    ("ecg", re.compile(r"\b(?:ecg|ekg|electrocardiogra(?:m|phy)|tracing)\b", re.I)),
)


def _modalities(value: str) -> frozenset[str]:
    return frozenset(name for name, pattern in _MODALITY_PATTERNS if pattern.search(value))


def deterministic_binding_authority(
    patient_finding: str,
    criterion_text: str,
    *,
    candidate_label: str,
    role: str,
    normalizer: object | None = None,
) -> DeterministicBindingAuthority:
    """Grant bypass authority only to closed, reproducible typed operations.

    Lexical coverage and UMLS identity remain useful for proposing a pair, but
    can never authorize this boundary by themselves.
    """
    assessment = assess_typed_binding(
        patient_finding, criterion_text, role=role, normalizer=normalizer
    )
    if not assessment.admissible or role != "diagnostic_criterion":
        return DeterministicBindingAuthority(False, "semantic_review", "typed safety checks did not establish an executable diagnostic binding")

    patient = assessment.patient_atom
    criterion = assessment.criterion_atom
    patient_tokens = set(patient.tokens)
    criterion_tokens = set(criterion.tokens)
    shared_subject = (patient_tokens & criterion_tokens) - POLARITY_ONLY_TOKENS

    if criterion.quantities and patient.quantities and shared_subject:
        return DeterministicBindingAuthority(True, "exact_measurement", "normalized measurement, unit, comparator, and threshold were executed")

    patient_modalities = _modalities(patient.text)
    criterion_modalities = _modalities(criterion.text)
    if patient_modalities or criterion_modalities:
        same_modality = bool(patient_modalities and patient_modalities == criterion_modalities)
        positive_completed = (
            patient.polarity == Polarity.PRESENT
            and patient.temporality != Temporality.HISTORICAL
            and bool(COMPLETED_RESULT_RE.search(patient.text))
            and not NONFINAL_TEST_RE.search(patient.text)
        )
        # A matching modality is insufficient without a shared result/anatomic
        # subject; this prevents, for example, coronary angiography from
        # satisfying an endoscopic gastrointestinal warrant.
        if same_modality and positive_completed and shared_subject:
            return DeterministicBindingAuthority(True, "exact_positive_test", "modality and asserted result/anatomic subject match exactly")
        return DeterministicBindingAuthority(False, "semantic_review", "test modality, result subject, or completion status is not an exact match")

    if patient.polarity == Polarity.ABSENT and criterion.polarity == Polarity.ABSENT and shared_subject:
        return DeterministicBindingAuthority(True, "explicit_negation", "matching clinical subject is explicitly absent")

    candidate = clinical_atom(candidate_label, normalizer)
    candidate_tokens = set(candidate.tokens)
    candidate_named = bool(candidate_tokens) and candidate_tokens <= patient_tokens
    explicit_assertion = bool(re.search(
        r"\b(?:diagnos(?:ed|is)|has|have|with|confirmed|present|current|acute)\b",
        patient.text,
        re.IGNORECASE,
    ))
    if (
        candidate_named
        and explicit_assertion
        and patient.polarity == Polarity.PRESENT
        and patient.temporality != Temporality.HISTORICAL
    ):
        return DeterministicBindingAuthority(True, "explicit_current_diagnosis", "the current finding explicitly asserts the candidate")

    return DeterministicBindingAuthority(False, "semantic_review", "match depends on lexical or terminology similarity rather than an executable operation")


def _tokens(value: str) -> tuple[str, ...]:
    output = []
    for raw in TOKEN_RE.findall(value):
        token = raw.casefold().strip("-/")
        token = re.sub(r"[-/:][<>]?\d.*$", "", token)
        compact = re.sub(r"[^a-z0-9]", "", token)
        if compact in {
            "ctn", "ctnt", "ctropnt", "hsctn", "hsctnt", "tropt",
            "troponint",
        }:
            token = "troponin"
        if len(token) > 5 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 5 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        if len(token) >= 3 and token not in STOPWORDS:
            output.append(token)
    return tuple(output)


def _rising_abnormal_laboratory_match(
    patient_text: str,
    patient_tokens: set[str],
    criterion_tokens: set[str],
) -> bool:
    """Recognize an explicit serial rise in one abnormal-flagged assay.

    The asterisk is not interpreted alone because it does not encode the
    direction of abnormality. Admission requires at least two values for the
    same named assay, a final value above the first, a shared assay concept,
    and an elevation predicate in the criterion.
    """
    if not (criterion_tokens & ELEVATION_TOKENS):
        return False
    series: dict[str, list[float]] = {}
    for match in ABNORMAL_LAB_RE.finditer(patient_text):
        name_tokens = _tokens(match.group("name"))
        if len(name_tokens) != 1:
            continue
        name = name_tokens[0]
        if name not in patient_tokens or name not in criterion_tokens:
            continue
        series.setdefault(name, []).append(float(match.group("value")))
    return any(
        len(values) >= 2 and values[-1] > values[0]
        for values in series.values()
    )


def patient_atomic_spans(value: str) -> tuple[str, ...]:
    """Return sentence atoms plus exact bounded serial-laboratory substrings."""
    spans = [
        span
        for raw in PATIENT_CLAUSE_RE.split(value)
        if len((span := " ".join(raw.split()))) >= 8
    ]
    lab_matches: dict[str, list[re.Match[str]]] = {}
    for match in ABNORMAL_LAB_RE.finditer(value):
        name_tokens = _tokens(match.group("name"))
        if len(name_tokens) == 1:
            lab_matches.setdefault(name_tokens[0], []).append(match)
    for matches in lab_matches.values():
        if len(matches) < 2:
            continue
        serial_span = " ".join(
            value[matches[0].start():matches[-1].end()].split()
        )
        if 8 <= len(serial_span) <= 2000:
            spans.append(serial_span)
    return tuple(dict.fromkeys(spans))


def _phrases(tokens: tuple[str, ...]) -> set[tuple[str, ...]]:
    return {
        tuple(tokens[index:index + size])
        for size in (2, 3, 4)
        for index in range(len(tokens) - size + 1)
    }


def _morphologically_compatible(left: str, right: str) -> bool:
    """Allow conservative inflection/derivation matching of long medical terms."""
    if left == right:
        return True
    if min(len(left), len(right)) < 6:
        return False
    prefix = 0
    for a, b in zip(left, right):
        if a != b:
            break
        prefix += 1
    return prefix >= 5 and max(len(left), len(right)) - prefix <= 4


def _quantities(value: str) -> tuple[Quantity, ...]:
    output = []
    for match in NUMBER_RE.finditer(value):
        raw_operator = (match.group("operator") or "=").casefold()
        operator = {
            "below": "<", "under": "<", "above": ">", "over": ">",
            "more than": ">", "less than": "<", "greater than": ">",
            "exceeding": ">", "exceeds": ">", "exceeded": ">",
            "surpassing": ">", "surpasses": ">", "surpassed": ">",
            "at least": ">=", "at most": "<=",
            "≥": ">=", "≤": "<=",
        }.get(raw_operator, raw_operator)
        output.append(Quantity(
            value=float(match.group("value")),
            unit=(match.group("unit") or "").casefold(),
            operator=operator,
        ))
    return tuple(output)


def _cuis(value: str, normalizer: object | None) -> tuple[str, ...]:
    if normalizer is None:
        return ()
    words = TOKEN_RE.findall(value)
    terms = list(extract_candidate_terms(value, limit=16))
    terms.extend(
        word for word in words
        if len(word) >= 2 and word.upper() == word and any(ch.isalpha() for ch in word)
    )
    terms.extend(
        " ".join(words[index:index + size])
        for size in (2, 3, 4)
        for index in range(len(words) - size + 1)
    )
    found = []
    for term in dict.fromkeys(terms):
        concept = normalizer.concept(term)
        if concept is not None and concept.cui not in found:
            found.append(concept.cui)
    return tuple(found)


def clinical_atom(value: str, normalizer: object | None = None) -> ClinicalAtom:
    historical = bool(HISTORICAL_RE.search(value))
    current = bool(CURRENT_RE.search(value) or CURRENT_COMPARISON_RE.search(value))
    temporality = (
        Temporality.CURRENT if current
        else Temporality.HISTORICAL if historical
        else Temporality.UNSPECIFIED
    )
    return ClinicalAtom(
        text=" ".join(value.split()),
        tokens=_tokens(value),
        cuis=_cuis(value, normalizer),
        polarity=Polarity.ABSENT if NEGATION_RE.search(value) else Polarity.PRESENT,
        temporality=temporality,
        quantities=_quantities(value),
    )


def assess_family_diagnostic_test_result(
    patient_finding: str,
    family_label: str,
    *,
    normalizer: object | None = None,
) -> DiagnosticTestResultAssessment:
    """Recognize a completed positive test that directly names a graph family.

    This certificate is family-only. It must not be reused to select a child or
    to satisfy subtype-specific conjuncts.
    """
    patient = clinical_atom(patient_finding, normalizer)
    family = clinical_atom(family_label, normalizer)
    test_present = bool(DIAGNOSTIC_TEST_RE.search(patient.text))
    completed = bool(COMPLETED_RESULT_RE.search(patient.text))
    nonfinal = bool(NONFINAL_TEST_RE.search(patient.text))
    current = patient.temporality != Temporality.HISTORICAL
    positive = patient.polarity == Polarity.PRESENT
    shared_cuis = set(patient.cuis) & set(family.cuis)
    family_tokens = set(family.tokens)
    patient_tokens = set(patient.tokens)
    matched_family_tokens = {
        family_token for family_token in family_tokens
        if any(
            _morphologically_compatible(family_token, patient_token)
            for patient_token in patient_tokens
        )
    }
    lexical_fraction = len(matched_family_tokens) / max(len(family_tokens), 1)
    concept_match = bool(shared_cuis) or lexical_fraction >= 0.75
    completed_ok = completed and not nonfinal
    questions = (
        CriticalQuestionResult("observation_present", positive, patient.polarity.value),
        CriticalQuestionResult("concept_compatible", concept_match, "family CUI or specific label coverage"),
        CriticalQuestionResult("criterion_satisfied", test_present and completed_ok and concept_match, "atomic criterion coverage=1.000; family-only completed test result"),
        CriticalQuestionResult("temporality_compatible", current, patient.temporality.value),
        CriticalQuestionResult("quantity_compatible", True, "not applicable to categorical test result"),
        CriticalQuestionResult("scheme_role_compatible", True, "diagnostic_criterion"),
        CriticalQuestionResult("diagnostic_test_present", test_present, "named diagnostic test"),
        CriticalQuestionResult("test_result_completed", completed_ok, "completed versus ordered/equivocal"),
        CriticalQuestionResult("family_only_scope", True, "not valid for child selection"),
    )
    admissible = all(item.passed for item in questions)
    score = 1.0 if admissible else max(lexical_fraction, 0.0)
    return DiagnosticTestResultAssessment(admissible, round(score, 6), questions)


def atomic_criteria(value: str, normalizer: object | None = None) -> tuple[ClinicalAtom, ...]:
    atoms = tuple(
        clinical_atom(clause, normalizer)
        for raw in CLAUSE_RE.split(value)
        if (clause := " ".join(raw.split()))
    )
    return atoms or (clinical_atom(value, normalizer),)


def _quantity_compatible(patient: ClinicalAtom, criterion: ClinicalAtom) -> bool:
    ranges = tuple(RANGE_RE.finditer(criterion.text))
    if ranges:
        for interval in ranges:
            low = float(interval.group("low"))
            high = float(interval.group("high"))
            unit = (interval.group("unit") or "").casefold()
            for observed in patient.quantities:
                if observed.operator != "=":
                    continue
                if unit and observed.unit != unit:
                    continue
                if low <= observed.value <= high:
                    return True
        return False
    thresholds = tuple(q for q in criterion.quantities if q.operator != "=")
    if not thresholds:
        return True
    for threshold in thresholds:
        for observed in patient.quantities:
            if threshold.unit and observed.unit != threshold.unit:
                continue
            if threshold.operator == ">" and observed.value > threshold.value:
                return True
            if threshold.operator == ">=" and observed.value >= threshold.value:
                return True
            if threshold.operator == "<" and observed.value < threshold.value:
                return True
            if threshold.operator == "<=" and observed.value <= threshold.value:
                return True
    return False


def assess_typed_binding(
    patient_finding: str,
    criterion_text: str,
    *,
    role: str,
    normalizer: object | None = None,
) -> TypedBindingAssessment:
    patient = clinical_atom(patient_finding, normalizer)
    best: tuple[float, ClinicalAtom, bool, bool] | None = None
    for criterion in atomic_criteria(criterion_text, normalizer):
        patient_tokens = set(patient.tokens)
        criterion_tokens = set(criterion.tokens)
        shared = patient_tokens & criterion_tokens
        lexical_coverage = len(shared) / max(len(criterion_tokens), 1)
        shared_cuis = set(patient.cuis) & set(criterion.cuis)
        concept_match = bool(shared_cuis)
        concept_coverage = (
            len(shared_cuis) / len(set(criterion.cuis))
            if criterion.cuis else 0.0
        )
        phrase_match = bool(_phrases(patient.tokens) & _phrases(criterion.tokens))
        specific_match = any(
            "-" in token or "/" in token or any(ch.isdigit() for ch in token)
            for token in shared
        )
        # A bare value on either side ("0.02", "the 99th percentile") states
        # no relationship to normal/abnormal by itself. This branch is only
        # trustworthy when at least one side states an explicit direction:
        # either the criterion carries a real, checkable threshold (in which
        # case quantity_compatible below independently verifies the patient's
        # value against it, even if the patient's own statement is a bare
        # value -- e.g. "BP is 170/100" against "above 140/90"), or the
        # criterion's own number is not a checkable threshold at all (e.g.
        # "exceeded the 99th percentile of the normal control value", whose
        # comparator word is not adjacent to a real number and so parses as
        # "="), in which case only the patient's own stated direction
        # ("greater than 8") is evidence of an abnormal reading -- a bare
        # patient value there is not.
        numeric_comparison = bool(criterion.quantities) and (
            any(q.operator != "=" for q in criterion.quantities)
            or any(q.operator != "=" for q in patient.quantities)
        )
        rising_abnormal_lab = (
            len(criterion_tokens) <= 8
            and _rising_abnormal_laboratory_match(
                patient.text,
                patient_tokens,
                criterion_tokens,
            )
        )
        non_st_elevation_match = bool(
            NON_ST_ELEVATION_RE.search(criterion.text)
            and ST_DEPRESSION_RE.search(patient.text)
        )
        completed_diagnostic_test_match = bool(
            role == "diagnostic_criterion"
            and patient.polarity == Polarity.PRESENT
            and patient.temporality != Temporality.HISTORICAL
            and DIAGNOSTIC_TEST_RE.search(patient.text)
            and COMPLETED_RESULT_RE.search(patient.text)
            and not NONFINAL_TEST_RE.search(patient.text)
            and DIAGNOSTIC_TEST_RE.search(criterion.text)
            and (concept_match or phrase_match or lexical_coverage >= 0.4)
        )
        # UMLS establishes terminology compatibility, not clinical entailment.
        # It may close a synonym gap only when the atomic criterion is itself a
        # short concept assertion; it must never make a long rule fully true.
        short_concept_assertion = (
            concept_coverage == 1.0
            and len(criterion_tokens) <= 4
            and not criterion.quantities
        )
        score = max(
            lexical_coverage,
            1.0 if short_concept_assertion else 0.0,
            1.0 if non_st_elevation_match else 0.0,
        )
        absence_match = (
            patient.polarity == Polarity.ABSENT
            and criterion.polarity == Polarity.ABSENT
            and bool(shared - POLARITY_ONLY_TOKENS)
            and score >= 0.4
        )
        semantic_match = (
            concept_match
            or phrase_match
            or specific_match
            or score >= 0.6
            or absence_match
            or rising_abnormal_lab
            or non_st_elevation_match
            or completed_diagnostic_test_match
        )
        criterion_covered = score >= 0.6 or (
            # A shared quantity on both sides plus a literal shared phrase
            # (e.g. "peak troponin") already pins the same named assay and
            # comparison; requiring the *whole* criterion sentence's
            # boilerplate ("...of the normal control value") to also
            # lexically overlap only punishes verbose KG prose. The actual
            # numeric agreement is independently re-checked by
            # quantity_compatible below, so this branch cannot admit a
            # numerically wrong pair on its own.
            numeric_comparison and phrase_match
        ) or (
            absence_match
        ) or (
            rising_abnormal_lab
            or non_st_elevation_match
            or completed_diagnostic_test_match
        )
        record = (score, criterion, semantic_match, criterion_covered)
        if best is None or record[0] > best[0]:
            best = record
    assert best is not None
    score, criterion, semantic_match, criterion_covered = best
    polarity_ok = patient.polarity == criterion.polarity
    temporality_ok = not (
        role == "diagnostic_criterion"
        and patient.temporality == Temporality.HISTORICAL
    )
    quantity_ok = _quantity_compatible(patient, criterion)
    role_ok = role in {
        "diagnostic_criterion", "clinical_feature", "risk_factor", "guideline"
    }
    if role == "risk_factor":
        criterion_covered = criterion_covered or score >= 0.4
    questions = (
        CriticalQuestionResult("observation_present", polarity_ok, patient.polarity.value),
        CriticalQuestionResult("concept_compatible", semantic_match, "UMLS/phrase/specific-token match"),
        CriticalQuestionResult("criterion_satisfied", criterion_covered, f"atomic criterion coverage={score:.3f}"),
        CriticalQuestionResult("temporality_compatible", temporality_ok, patient.temporality.value),
        CriticalQuestionResult("quantity_compatible", quantity_ok, "operator/unit comparison"),
        CriticalQuestionResult("scheme_role_compatible", role_ok, role),
    )
    admissible = all(item.passed for item in questions)
    return TypedBindingAssessment(
        admissible=admissible,
        score=round(score, 6),
        matched_clause=criterion.text,
        patient_atom=patient,
        criterion_atom=criterion,
        critical_questions=questions,
    )
