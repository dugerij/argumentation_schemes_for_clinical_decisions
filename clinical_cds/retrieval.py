"""Case-rendering and patient-evidence utilities.

The knowledge-graph retriever classes that used to live in this module
(``KnowledgeRetriever``, ``CandidateFirstKnowledgeRetriever``,
``SemanticRoutedKnowledgeRetriever``, ``build_knowledge_retriever``) were
removed as dead code: the executed pipeline retrieves knowledge-graph
evidence through ``graphrag_runtime.retrieval.FixedGraphRagKnowledgeRetriever``
instead. ``tokenize``, ``patient_evidence``, and ``render_case`` below remain
live and are imported by ``clinical_cds.runner``, ``clinical_cds.experiment``,
``clinical_cds.evaluation``, and ``graphrag_runtime.retrieval``.
"""

from __future__ import annotations

import re

from clinical_cds.schema import ClinicalCase, RetrievalBundle


TOKEN_RE = re.compile(r"[a-z0-9]+")
IMAGING_TERM_RE = re.compile(
    r"\b(?:x[- ]?ray|radiograph(?:y|ic)?|radiolog(?:y|ic|ist)?|"
    r"ct(?:\s+scan)?|computed tomography|mri|magnetic resonance|"
    r"ultrasound|sonograph(?:y|ic)?|echocardiograph(?:y|ic)?|"
    r"imaging|scan)\b",
    flags=re.IGNORECASE,
)
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+|[\r\n]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "patient",
    "she",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "with",
}
SECTION_EVIDENCE_IDS = {
    "chief_complaint": "S-CC",
    "history_of_present_illness": "S-HPI",
    "past_medical_history": "S-PMH",
    "family_history": "S-FH",
    "physical_exam": "S-PE",
    "pertinent_results": "S-RESULTS",
    "patient_info": "S-DEMOGRAPHICS",
    "initial_vitals": "S-VITALS",
    "tests": "S-TESTS",
    "question": "S-QUESTION",
}
# Retained for backward compatibility with clinical_cds.runner, which checks
# a live GraphRAG retriever's retriever_id against these version strings.
CANDIDATE_FIRST_RETRIEVER_VERSION = "candidate-first-section-rrf-v1"
SEMANTIC_ROUTED_RETRIEVER_VERSION = "semantic-routed-candidate-first-v1"


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in TOKEN_RE.findall(text.casefold())
        if len(token) > 1 and token not in STOPWORDS
    )


def section_evidence(case: ClinicalCase) -> tuple[tuple[str, str, str], ...]:
    output: list[tuple[str, str, str]] = []
    used_ids: set[str] = set()
    for index, (section_name, text) in enumerate(case.sections.items(), start=1):
        if not text.strip():
            continue
        evidence_id = SECTION_EVIDENCE_IDS.get(section_name, f"S-OTHER-{index}")
        if evidence_id in used_ids:
            evidence_id = f"{evidence_id}-{index}"
        used_ids.add(evidence_id)
        output.append((evidence_id, section_name, text))
    return tuple(output)


def imaging_evidence(case: ClinicalCase) -> tuple[tuple[str, str, str], ...]:
    excerpts: list[tuple[str, str, str]] = []
    for section_name, text in case.sections.items():
        matches = [
            sentence.strip()
            for sentence in SENTENCE_BOUNDARY_RE.split(text)
            if sentence.strip() and IMAGING_TERM_RE.search(sentence)
        ]
        for sentence in matches[:2]:
            evidence_id = f"S-IMG-{len(excerpts) + 1:02d}"
            excerpts.append((evidence_id, section_name, sentence[:600]))
            if len(excerpts) == 6:
                return tuple(excerpts)
    return tuple(excerpts)


def patient_evidence(case: ClinicalCase) -> tuple[tuple[str, str, str], ...]:
    return section_evidence(case) + imaging_evidence(case)


def render_case(case: ClinicalCase) -> str:
    lines = [
        f"{evidence_id} | {section_name.replace('_', ' ').title()} | {text}"
        for evidence_id, section_name, text in section_evidence(case)
    ]
    excerpts = imaging_evidence(case)
    if excerpts:
        lines.append("Imaging evidence excerpts:")
        lines.extend(
            (
                f"{evidence_id} | {section_name.replace('_', ' ').title()} "
                f"(imaging excerpt) | {text}"
            )
            for evidence_id, section_name, text in excerpts
        )
    if case.options:
        lines.append("Answer options:")
        lines.extend(f"{key}. {value}" for key, value in case.options.items())
    return "\n".join(lines)


def render_flat_retrieval(bundle: RetrievalBundle) -> str:
    if not bundle.facts:
        return "No guideline premises were retrieved."
    return "\n".join(
        (
            f"{fact.evidence_id} | diagnosis={fact.diagnosis_label} | "
            f"premise_type={fact.premise_type} | premise={fact.text}"
        )
        for fact in bundle.facts
    )


def render_graph_retrieval(bundle: RetrievalBundle) -> str:
    if not bundle.facts:
        return "No guideline subgraph was retrieved."
    lines: list[str] = []
    for fact in bundle.facts:
        path = " -> ".join(fact.diagnostic_path)
        lines.append(
            f"{fact.evidence_id} --supports--> {fact.diagnosis_label} "
            f"| type={fact.premise_type} | premise={fact.text}"
        )
        lines.append(f"PATH {fact.evidence_id}: {path}")
    return "\n".join(lines)
