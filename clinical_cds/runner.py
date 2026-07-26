from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from tqdm.auto import tqdm

from clinical_cds.argumentation import (
    REASONER_OUTPUT_SCHEMA,
    ArgumentScheme,
    PatientArgumentGraph,
    ReasonerProposal,
    SymbolicResolution,
    VerifierReport,
    build_patient_argument_graph,
    parse_reasoner_proposal,
    parse_verifier_report,
    render_proposal_for_verification,
    render_resolution_trace,
    resolve_argument_graph,
    verifier_output_schema,
)
from clinical_cds.io import append_jsonl
from clinical_cds.model import DiagnosticModel, OUTPUT_SCHEMA
from clinical_cds.retrieval import (
    KnowledgeRetriever,
    render_case,
    render_flat_retrieval,
    render_graph_retrieval,
    section_evidence,
)
from clinical_cds.schema import (
    ClinicalCase,
    ExperimentMode,
    PredictedObservation,
    PredictionRecord,
    RetrievalBundle,
)


PROMPT_VERSION = "diagnostic-symbolic-v2"
STANDARD_SYSTEM_PROMPT = """You are evaluating a diagnostic reasoning method.
Determine the single most likely diagnosis from the submitted clinical state
and supplied evidence.
Return one JSON object that follows the requested schema.
Citations must use the supplied S or K identifiers.
Set abstain to true unless the evidence supports a defensible diagnosis."""
REASONER_SYSTEM_PROMPT = """You are the diagnostic reasoner agent in a formal
argumentation experiment. Propose at most three candidate diagnoses and express
their support as at most two strong, non-duplicate argument-scheme instances per
diagnosis. Use only supplied patient and guideline evidence identifiers. Risk
factors may adjust plausibility but may not serve as diagnostic proof. Do not
resolve attacks and do not use unstated facts. Return only one JSON object
following the supplied schema."""
VERIFIER_SYSTEM_PROMPT = """You are the independent diagnostic verifier agent.
Evaluate every proposed argument against the submitted patient state, retrieved
guideline evidence, and its scheme-specific critical questions. You may support,
undercut, or mark an argument uncertain and may add evidence-grounded rebuttals
or undercutters. Do not use a gold answer and do not select the final diagnosis.
Return only one JSON object following the supplied schema."""
ARGUMENT_MODES = {
    ExperimentMode.STRUCTURED_ARGUMENT,
    ExperimentMode.SYMBOLIC_ARGUMENT,
}


@dataclass(frozen=True)
class ExperimentRun:
    run_id: str
    output_dir: Path
    predictions_path: Path
    argument_traces_path: Path
    manifest_path: Path
    records: tuple[PredictionRecord, ...]


@dataclass(frozen=True)
class CachedCompletion:
    response: str
    latency_seconds: float
    cache_hit: bool
    prompt_hash: str


@dataclass(frozen=True)
class ArgumentTeamResult:
    trace_id: str
    case_id: str
    proposal: ReasonerProposal | None
    verifier: VerifierReport | None
    graph: PatientArgumentGraph | None
    resolution: SymbolicResolution | None
    reasoner_prompt_hash: str
    verifier_prompt_hash: str
    latency_seconds: float
    cache_hit: bool
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "case_id": self.case_id,
            "proposal": self.proposal.to_dict() if self.proposal else None,
            "verifier": self.verifier.to_dict() if self.verifier else None,
            "argument_graph": self.graph.to_dict() if self.graph else None,
            "symbolic_resolution": (
                self.resolution.to_dict() if self.resolution else None
            ),
            "human_readable_trace": (
                render_resolution_trace(self.graph, self.resolution)
                if self.graph and self.resolution
                else ""
            ),
            "reasoner_prompt_hash": self.reasoner_prompt_hash,
            "verifier_prompt_hash": self.verifier_prompt_hash,
            "latency_seconds": self.latency_seconds,
            "cache_hit": self.cache_hit,
            "error": self.error,
        }


def _model_manifest(model: DiagnosticModel) -> dict[str, Any]:
    decoding_config = getattr(model, "decoding_config", {})
    return {
        "model_id": model.model_id,
        "decoding": (
            dict(decoding_config)
            if isinstance(decoding_config, dict)
            else {}
        ),
    }


class ResponseCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)

    def _path(self, prompt_hash: str) -> Path:
        return self.cache_dir / prompt_hash[:2] / f"{prompt_hash}.json"

    def get(self, prompt_hash: str) -> tuple[str, float] | None:
        path = self._path(prompt_hash)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return (
            str(payload["response"]),
            float(payload.get("latency_seconds") or 0.0),
        )

    def put(
        self,
        prompt_hash: str,
        response: str,
        model_id: str,
        latency_seconds: float,
        role: str,
    ) -> None:
        path = self._path(prompt_hash)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "prompt_hash": prompt_hash,
                    "model_id": model_id,
                    "prompt_version": PROMPT_VERSION,
                    "role": role,
                    "response": response,
                    "latency_seconds": latency_seconds,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        temporary.replace(path)


def _first_json_object(value: str) -> dict[str, Any]:
    normalized = value.replace("```json", "").replace("```", "").strip()
    try:
        payload = json.loads(normalized)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    start = normalized.find("{")
    if start < 0:
        raise ValueError("Model output did not contain a JSON object.")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(normalized)):
        character = normalized[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                payload = json.loads(normalized[start : index + 1])
                if not isinstance(payload, dict):
                    raise ValueError("Model output JSON must be an object.")
                return payload
    raise ValueError("Model output contained an incomplete JSON object.")


def _string_list(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    values = value if isinstance(value, list) else [value]
    output: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = " ".join(str(item).split()).strip()
        if text and text.casefold() not in seen:
            seen.add(text.casefold())
            output.append(text)
    return tuple(output)


def _parse_observations(value: object) -> tuple[PredictedObservation, ...]:
    if not isinstance(value, list):
        return ()
    observations: list[PredictedObservation] = []
    for item in value:
        if isinstance(item, dict):
            text = " ".join(str(item.get("text") or "").split()).strip()
            source_id = " ".join(str(item.get("source_id") or "").split()).strip() or None
        else:
            text = " ".join(str(item).split()).strip()
            source_id = None
        if text:
            observations.append(
                PredictedObservation(text=text, source_id=source_id)
            )
    return tuple(observations)


def _normalize_answer(answer: object, case: ClinicalCase) -> str:
    text = " ".join(str(answer or "").split()).strip()
    option_key = text.upper().strip(" .():")
    if case.options and option_key in case.options:
        return case.options[option_key]
    if case.options:
        for option in case.options.values():
            if text.casefold().strip(" .") == option.casefold().strip(" ."):
                return option

    diagnostic_prefixes = (
        r"^(?:answer|diagnosis)\s*:\s*",
        r"^(?:the\s+)?(?:most\s+likely\s+)?diagnosis\s+is\s+",
        r"^(?:the\s+)?patient(?:'s)?\s+clinical\s+presentation\s+is\s+"
        r"(?:most\s+)?consistent\s+with\s+",
        r"^(?:the\s+)?clinical\s+presentation\s+is\s+"
        r"(?:most\s+)?consistent\s+with\s+",
    )
    for pattern in diagnostic_prefixes:
        normalized = re.sub(pattern, "", text, flags=re.IGNORECASE)
        if normalized != text:
            text = normalized
            break
    text = re.split(r"\s+(?:based on|because|given)\s+", text, maxsplit=1)[0]
    return text.strip(" .")


def _retrieved_metadata(bundle: RetrievalBundle) -> list[dict[str, Any]]:
    return [
        {
            "evidence_id": fact.evidence_id,
            "node_id": fact.node_id,
            "category": fact.category,
            "diagnosis_label": fact.diagnosis_label,
            "premise_type": fact.premise_type,
            "score": fact.score,
            "diagnostic_path": list(fact.diagnostic_path),
        }
        for fact in bundle.facts
    ]


def _build_standard_prompt(
    case: ClinicalCase,
    mode: ExperimentMode,
    bundle: RetrievalBundle,
) -> tuple[str, tuple[str, ...]]:
    task_instruction = (
        "Select the single best answer from the supplied options."
        if case.options
        else "State the single most likely diagnosis."
    )
    sections = [
        f"Task: {task_instruction}",
        "",
        "Clinical state:",
        render_case(case),
    ]
    if mode == ExperimentMode.FLAT_RAG:
        sections.extend(
            [
                "",
                "Retrieved guideline premises (independent flat records):",
                render_flat_retrieval(bundle),
            ]
        )
    elif mode == ExperimentMode.GRAPH_RAG:
        sections.extend(
            [
                "",
                "Retrieved guideline subgraph:",
                render_graph_retrieval(bundle),
            ]
        )
    elif mode != ExperimentMode.DIRECT:
        raise ValueError(f"Standard prediction does not support mode {mode.value}.")

    section_ids = tuple(item[0] for item in section_evidence(case))
    knowledge_ids = bundle.evidence_ids if mode != ExperimentMode.DIRECT else ()
    valid_ids = section_ids + knowledge_ids
    sections.extend(
        [
            "",
            (
                "Return JSON with: answer, concise reasoning of at most three sentences, "
                "at most four citations, at most three observations "
                "(each with text and source_id), and abstain."
            ),
            (
                "The answer value must contain only the diagnosis or selected "
                "option text. Citations and observation source_id values must "
                "use only these exact identifiers: "
                + ", ".join(valid_ids)
            ),
        ]
    )
    return "\n".join(sections), valid_ids


def _build_reasoner_prompt(
    case: ClinicalCase,
    bundle: RetrievalBundle,
) -> tuple[str, tuple[str, ...]]:
    section_ids = tuple(item[0] for item in section_evidence(case))
    valid_ids = section_ids + bundle.evidence_ids
    option_instruction = (
        "Candidate diagnoses must use the exact supplied option text."
        if case.options
        else "Candidate diagnoses must be concise diagnostic labels."
    )
    prompt = "\n".join(
        [
            "Task: construct a patient-specific diagnostic argument set.",
            option_instruction,
            "",
            "Clinical state:",
            render_case(case),
            "",
            "Retrieved diagnostic guideline graph:",
            render_graph_retrieval(bundle),
            "",
            "Permitted support schemes:",
            (
                "- argument_from_clinical_sign: a patient finding and guideline "
                "warrant support a diagnosis."
            ),
            (
                "- argument_from_diagnostic_criterion: a patient result satisfies "
                "a retrieved diagnostic criterion."
            ),
            (
                "- argument_from_risk_factor: history adjusts plausibility but "
                "does not prove a diagnosis."
            ),
            (
                "- argument_from_guideline_authority: a retrieved guideline premise "
                "provides an applicable warrant."
            ),
            "",
            (
                "Every clinical-sign, diagnostic-criterion, or risk-factor argument "
                "must cite at least one S patient identifier and one K guideline identifier."
            ),
            (
                "Guideline-authority arguments must cite at least one K identifier. "
                "Use only these identifiers: "
                + ", ".join(valid_ids)
            ),
            (
                "Use at most two strongest, non-duplicate arguments for each "
                "candidate diagnosis."
            ),
        ]
    )
    return prompt, valid_ids


def _build_verifier_prompt(
    case: ClinicalCase,
    bundle: RetrievalBundle,
    proposal: ReasonerProposal,
) -> str:
    return "\n".join(
        [
            "Task: verify the structured diagnostic arguments.",
            "",
            "Clinical state:",
            render_case(case),
            "",
            "Retrieved diagnostic guideline graph:",
            render_graph_retrieval(bundle),
            "",
            "Proposed arguments and applicable critical questions:",
            render_proposal_for_verification(proposal),
            "",
            (
                "Return exactly one review for every A argument identifier. "
                "A supported verdict requires explicit evidence and no failed "
                "critical question."
            ),
            (
                "Return reviews in A-identifier order. Keep each explanation to "
                "one sentence of at most 20 words."
            ),
            (
                "Counterarguments may target an A support argument or D diagnosis "
                "argument. They must use only supplied evidence identifiers."
            ),
        ]
    )


def _argument_observations(
    arguments: Iterable[Any],
    valid_ids: set[str],
) -> tuple[PredictedObservation, ...]:
    observations: list[PredictedObservation] = []
    for argument in arguments:
        source_id = next(
            (
                evidence_id
                for evidence_id in argument.evidence_ids
                if evidence_id.startswith("S-") and evidence_id in valid_ids
            ),
            None,
        )
        if source_id is None:
            continue
        observations.append(
            PredictedObservation(
                text=argument.premise,
                source_id=source_id,
            )
        )
        if len(observations) == 3:
            break
    return tuple(observations)


def _argument_citations(
    arguments: Iterable[Any],
    valid_ids: set[str],
) -> tuple[str, ...]:
    return tuple(
        list(
            dict.fromkeys(
                evidence_id
                for argument in arguments
                for evidence_id in argument.evidence_ids
                if evidence_id in valid_ids
            )
        )[:4]
    )


class ExperimentRunner:
    def __init__(
        self,
        *,
        model: DiagnosticModel,
        retriever: KnowledgeRetriever,
        cache_dir: Path,
        top_k: int = 6,
        reasoner_model: DiagnosticModel | None = None,
        verifier_model: DiagnosticModel | None = None,
    ):
        self.model = model
        self.reasoner_model = reasoner_model or model
        self.verifier_model = verifier_model or model
        self.retriever = retriever
        self.cache = ResponseCache(cache_dir)
        self.top_k = top_k

    def _assert_data_boundary(self, case: ClinicalCase) -> None:
        checked: set[int] = set()
        for model in (
            self.model,
            self.reasoner_model,
            self.verifier_model,
        ):
            if id(model) in checked:
                continue
            checked.add(id(model))
            boundary_check = getattr(model, "assert_data_boundary", None)
            if callable(boundary_check):
                boundary_check(case.dataset)

    def _complete(
        self,
        *,
        role: str,
        model: DiagnosticModel,
        system_prompt: str,
        user_prompt: str,
        output_schema: dict[str, object],
    ) -> CachedCompletion:
        schema_text = json.dumps(output_schema, sort_keys=True, separators=(",", ":"))
        prompt_hash = hashlib.sha256(
            (
                PROMPT_VERSION
                + "\0"
                + role
                + "\0"
                + getattr(model, "cache_identity", model.model_id)
                + "\0"
                + schema_text
                + "\0"
                + system_prompt
                + "\0"
                + user_prompt
            ).encode("utf-8")
        ).hexdigest()
        cached = self.cache.get(prompt_hash)
        if cached is not None:
            response, latency_seconds = cached
            return CachedCompletion(
                response=response,
                latency_seconds=latency_seconds,
                cache_hit=True,
                prompt_hash=prompt_hash,
            )

        started = time.perf_counter()
        response = model.complete(
            system_prompt,
            user_prompt,
            output_schema=output_schema,
        )
        latency_seconds = time.perf_counter() - started
        self.cache.put(
            prompt_hash,
            response,
            model.model_id,
            latency_seconds,
            role,
        )
        return CachedCompletion(
            response=response,
            latency_seconds=latency_seconds,
            cache_hit=False,
            prompt_hash=prompt_hash,
        )

    def predict(
        self,
        case: ClinicalCase,
        mode: ExperimentMode,
        *,
        run_id: str,
        bundle: RetrievalBundle | None = None,
    ) -> PredictionRecord:
        if mode in ARGUMENT_MODES:
            raise ValueError(
                "Argument modes require run_argument_team and predict_argument_mode."
            )
        self._assert_data_boundary(case)
        if bundle is None:
            bundle = (
                RetrievalBundle(facts=(), query_tokens=())
                if mode == ExperimentMode.DIRECT
                else self.retriever.retrieve(case, top_k=self.top_k)
            )
        user_prompt, valid_ids = _build_standard_prompt(case, mode, bundle)

        started = time.perf_counter()
        try:
            completion = self._complete(
                role=f"standard:{mode.value}",
                model=self.model,
                system_prompt=STANDARD_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                output_schema=OUTPUT_SCHEMA,
            )
            payload = _first_json_object(completion.response)
            predicted_label = _normalize_answer(payload.get("answer"), case)
            reasoning = " ".join(
                str(payload.get("reasoning") or "").split()
            ).strip()
            citations = _string_list(payload.get("citations"))
            observations = _parse_observations(payload.get("observations"))
            abstained = bool(payload.get("abstain")) or not predicted_label
            error = None
            latency_seconds = completion.latency_seconds
            cache_hit = completion.cache_hit
            prompt_hash = completion.prompt_hash
        except Exception as exc:
            predicted_label = ""
            reasoning = ""
            citations = ()
            observations = ()
            abstained = True
            error = f"{type(exc).__name__}: {exc}"
            latency_seconds = time.perf_counter() - started
            cache_hit = False
            prompt_hash = ""

        return PredictionRecord(
            run_id=run_id,
            case_id=case.case_id,
            dataset=case.dataset,
            task=case.task,
            mode=mode,
            model_id=self.model.model_id,
            gold_label=case.gold_label,
            predicted_label=predicted_label,
            reasoning=reasoning,
            citations=citations,
            observations=observations,
            abstained=abstained,
            latency_seconds=round(latency_seconds, 6),
            prompt_hash=prompt_hash,
            cache_hit=cache_hit,
            valid_evidence_ids=valid_ids,
            quality_flags=case.quality_flags,
            error=error,
            metadata={
                "retriever_id": self.retriever.retriever_id,
                "retrieved_facts": (
                    _retrieved_metadata(bundle)
                    if mode != ExperimentMode.DIRECT
                    else []
                ),
                "directory_label": case.directory_label,
                "disease_category": case.disease_category,
            },
        )

    def run_argument_team(
        self,
        case: ClinicalCase,
        bundle: RetrievalBundle,
        *,
        run_id: str,
        stage_callback: Callable[[str, str], None] | None = None,
    ) -> ArgumentTeamResult:
        self._assert_data_boundary(case)
        trace_id = f"{run_id}:{case.case_id}"
        proposal: ReasonerProposal | None = None
        verifier: VerifierReport | None = None
        graph: PatientArgumentGraph | None = None
        resolution: SymbolicResolution | None = None
        reasoner_hash = ""
        verifier_hash = ""
        completions: list[CachedCompletion] = []
        settled_stages: set[str] = set()
        started = time.perf_counter()

        def notify(stage: str, event: str) -> None:
            if stage_callback is not None:
                stage_callback(stage, event)

        try:
            reasoner_prompt, valid_ids = _build_reasoner_prompt(case, bundle)
            notify("reasoner", "started")
            try:
                reasoner_completion = self._complete(
                    role="argument_reasoner",
                    model=self.reasoner_model,
                    system_prompt=REASONER_SYSTEM_PROMPT,
                    user_prompt=reasoner_prompt,
                    output_schema=REASONER_OUTPUT_SCHEMA,
                )
            finally:
                settled_stages.add("reasoner")
                notify("reasoner", "completed")
            completions.append(reasoner_completion)
            reasoner_hash = reasoner_completion.prompt_hash
            proposal = parse_reasoner_proposal(
                _first_json_object(reasoner_completion.response),
                case,
            )
            if not proposal.candidates:
                raise ValueError("Reasoner produced no valid diagnosis candidates.")
            if not any(candidate.arguments for candidate in proposal.candidates):
                raise ValueError("Reasoner produced no valid support arguments.")

            verifier_prompt = _build_verifier_prompt(case, bundle, proposal)
            notify("verifier", "started")
            try:
                verifier_completion = self._complete(
                    role="argument_verifier",
                    model=self.verifier_model,
                    system_prompt=VERIFIER_SYSTEM_PROMPT,
                    user_prompt=verifier_prompt,
                    output_schema=verifier_output_schema(proposal),
                )
            finally:
                settled_stages.add("verifier")
                notify("verifier", "completed")
            completions.append(verifier_completion)
            verifier_hash = verifier_completion.prompt_hash
            verifier = parse_verifier_report(
                _first_json_object(verifier_completion.response),
                proposal,
            )
            graph = build_patient_argument_graph(
                case_id=case.case_id,
                proposal=proposal,
                verifier=verifier,
                valid_evidence_ids=valid_ids,
                knowledge_support={
                    fact.evidence_id: (
                        fact.diagnosis_label,
                        *fact.diagnostic_path,
                    )
                    for fact in bundle.facts
                },
            )
            resolution = resolve_argument_graph(graph, proposal, verifier)
            error = None
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            for stage in ("reasoner", "verifier"):
                if stage not in settled_stages:
                    settled_stages.add(stage)
                    notify(stage, "skipped")

        latency_seconds = (
            sum(completion.latency_seconds for completion in completions)
            if completions
            else time.perf_counter() - started
        )
        return ArgumentTeamResult(
            trace_id=trace_id,
            case_id=case.case_id,
            proposal=proposal,
            verifier=verifier,
            graph=graph,
            resolution=resolution,
            reasoner_prompt_hash=reasoner_hash,
            verifier_prompt_hash=verifier_hash,
            latency_seconds=round(latency_seconds, 6),
            cache_hit=bool(completions) and all(
                completion.cache_hit for completion in completions
            ),
            error=error,
        )

    def predict_argument_mode(
        self,
        case: ClinicalCase,
        mode: ExperimentMode,
        team: ArgumentTeamResult,
        bundle: RetrievalBundle,
        *,
        run_id: str,
    ) -> PredictionRecord:
        if mode not in ARGUMENT_MODES:
            raise ValueError(f"Not an argument mode: {mode.value}")
        valid_id_sequence = (
            tuple(item[0] for item in section_evidence(case))
            + bundle.evidence_ids
        )
        valid_ids = set(valid_id_sequence)
        proposal = team.proposal
        verifier = team.verifier
        graph = team.graph
        resolution = team.resolution

        selected_arguments: tuple[Any, ...] = ()
        if team.error or not proposal or not verifier or not graph or not resolution:
            predicted_label = ""
            reasoning = ""
            abstained = True
            error = team.error or "Argument team did not produce a complete trace."
        elif mode == ExperimentMode.STRUCTURED_ARGUMENT:
            predicted_label = (
                "" if proposal.abstain else proposal.preferred_diagnosis
            )
            abstained = proposal.abstain or not predicted_label
            candidate = next(
                (
                    item
                    for item in proposal.candidates
                    if item.diagnosis == proposal.preferred_diagnosis
                ),
                None,
            )
            selected_arguments = candidate.arguments if candidate else ()
            reasoning = (
                f"The reasoner preferred {predicted_label or 'no diagnosis'} from "
                f"{len(proposal.candidates)} structured candidates. "
                f"The verifier reviewed {len(verifier.reviews)} of "
                f"{sum(len(item.arguments) for item in proposal.candidates)} arguments; "
                "symbolic resolution is withheld in this ablation."
            )
            error = None
        else:
            predicted_label = resolution.selected_diagnosis
            abstained = resolution.abstained
            selected_candidate = next(
                (
                    item
                    for item in proposal.candidates
                    if item.diagnosis == predicted_label
                ),
                None,
            )
            selected_arguments = tuple(
                argument
                for argument in (selected_candidate.arguments if selected_candidate else ())
                if argument.argument_id in resolution.accepted_argument_ids
            )
            reasoning = resolution.trace[-1] if resolution.trace else ""
            error = None

        citations = _argument_citations(selected_arguments, valid_ids)
        observations = _argument_observations(selected_arguments, valid_ids)
        prompt_hash = hashlib.sha256(
            (
                team.reasoner_prompt_hash
                + "\0"
                + team.verifier_prompt_hash
                + "\0"
                + mode.value
            ).encode("utf-8")
        ).hexdigest()
        quality = graph.quality if graph else None
        all_arguments = (
            [
                argument
                for candidate in proposal.candidates
                for argument in candidate.arguments
            ]
            if proposal
            else []
        )
        accepted_count = (
            len(resolution.accepted_argument_ids) if resolution else 0
        )
        rejected_count = (
            len(resolution.rejected_argument_ids) if resolution else 0
        )
        undecided_count = (
            len(resolution.undecided_argument_ids) if resolution else 0
        )
        return PredictionRecord(
            run_id=run_id,
            case_id=case.case_id,
            dataset=case.dataset,
            task=case.task,
            mode=mode,
            model_id=self.model.model_id,
            gold_label=case.gold_label,
            predicted_label=predicted_label,
            reasoning=reasoning,
            citations=citations,
            observations=observations,
            abstained=abstained,
            latency_seconds=team.latency_seconds,
            prompt_hash=prompt_hash,
            cache_hit=team.cache_hit,
            valid_evidence_ids=valid_id_sequence,
            quality_flags=case.quality_flags,
            error=error,
            metadata={
                "retriever_id": self.retriever.retriever_id,
                "retrieved_facts": _retrieved_metadata(bundle),
                "directory_label": case.directory_label,
                "disease_category": case.disease_category,
                "argument_trace_id": team.trace_id,
                "argument_count": len(all_arguments),
                "argument_schemes": sorted(
                    {argument.scheme.value for argument in all_arguments}
                    | {ArgumentScheme.BEST_EXPLANATION.value}
                ),
                "argument_graph_node_count": len(graph.nodes) if graph else 0,
                "argument_graph_relation_count": (
                    len(graph.relations) if graph else 0
                ),
                "argument_schema_validity": (
                    quality.argument_schema_validity if quality else 0.0
                ),
                "argument_evidence_validity": (
                    quality.argument_evidence_validity if quality else 0.0
                ),
                "valid_evidence_reference_fraction": (
                    quality.valid_evidence_reference_fraction
                    if quality
                    else 0.0
                ),
                "verifier_review_coverage": (
                    quality.verifier_review_coverage if quality else 0.0
                ),
                "reasoner_preferred_diagnosis": (
                    proposal.preferred_diagnosis if proposal else ""
                ),
                "symbolic_selected_diagnosis": (
                    resolution.selected_diagnosis if resolution else ""
                ),
                "argument_resolution_changed": bool(
                    proposal
                    and resolution
                    and proposal.preferred_diagnosis
                    != resolution.selected_diagnosis
                ),
                "resolver_id": (
                    resolution.resolver_id if resolution else ""
                ),
                "argument_status_counts": {
                    "accepted": accepted_count,
                    "rejected": rejected_count,
                    "undecided": undecided_count,
                },
                "agent_roles": {
                    "reasoner": self.reasoner_model.model_id,
                    "verifier": self.verifier_model.model_id,
                },
                "shared_agent_exchange": True,
            },
        )


def run_experiment(
    *,
    cases: Iterable[ClinicalCase],
    modes: Iterable[ExperimentMode],
    runner: ExperimentRunner,
    output_dir: Path,
    run_name: str | None = None,
    fail_fast: bool = False,
    show_progress: bool = False,
) -> ExperimentRun:
    case_list = tuple(cases)
    mode_list = tuple(dict.fromkeys(modes))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = run_name or f"experiment_{timestamp}_{uuid.uuid4().hex[:8]}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = run_dir / "predictions.jsonl"
    argument_traces_path = run_dir / "argument_traces.jsonl"
    manifest_path = run_dir / "manifest.json"

    records: list[PredictionRecord] = []
    argument_trace_count = 0
    needs_retrieval = any(mode != ExperimentMode.DIRECT for mode in mode_list)
    needs_argument_team = any(mode in ARGUMENT_MODES for mode in mode_list)
    standard_modes = tuple(
        mode for mode in mode_list if mode not in ARGUMENT_MODES
    )
    stages_per_case = len(standard_modes) + (
        2 if needs_argument_team else 0
    )
    progress_total = len(case_list) * stages_per_case
    with tqdm(
        total=progress_total,
        desc="Experiment",
        unit="stage",
        dynamic_ncols=True,
        disable=not show_progress,
    ) as progress:
        for case_number, case in enumerate(case_list, start=1):
            case_progress = f"case {case_number}/{len(case_list)}"
            if needs_retrieval:
                progress.set_postfix_str(
                    f"{case_progress} | retrieval",
                    refresh=True,
                )
            shared_bundle = (
                runner.retriever.retrieve(case, top_k=runner.top_k)
                if needs_retrieval
                else RetrievalBundle(facts=(), query_tokens=())
            )

            def update_argument_progress(stage: str, event: str) -> None:
                progress.set_postfix_str(
                    f"{case_progress} | {stage}",
                    refresh=True,
                )
                if event in {"completed", "skipped"}:
                    progress.update(1)

            team = (
                runner.run_argument_team(
                    case,
                    shared_bundle,
                    run_id=run_id,
                    stage_callback=update_argument_progress,
                )
                if needs_argument_team
                else None
            )
            if team is not None:
                append_jsonl(argument_traces_path, team.to_dict())
                argument_trace_count += 1

            for mode in mode_list:
                if mode in ARGUMENT_MODES and team is not None:
                    record = runner.predict_argument_mode(
                        case,
                        mode,
                        team,
                        shared_bundle,
                        run_id=run_id,
                    )
                else:
                    progress.set_postfix_str(
                        f"{case_progress} | {mode.value}",
                        refresh=True,
                    )
                    try:
                        record = runner.predict(
                            case,
                            mode,
                            run_id=run_id,
                            bundle=shared_bundle,
                        )
                    finally:
                        progress.update(1)
                append_jsonl(predictions_path, record.to_dict())
                records.append(record)
                if fail_fast and record.error:
                    raise RuntimeError(
                        f"Experiment failed for {case.case_id}/{mode.value}: "
                        f"{record.error}"
                    )
        progress.set_postfix_str("complete", refresh=True)

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompt_version": PROMPT_VERSION,
        "model_id": runner.model.model_id,
        "model_configurations": {
            "standard": _model_manifest(runner.model),
            "reasoner": _model_manifest(runner.reasoner_model),
            "verifier": _model_manifest(runner.verifier_model),
        },
        "retriever_id": runner.retriever.retriever_id,
        "normalizer_id": (
            runner.retriever.normalizer.normalizer_id
            if runner.retriever.normalizer is not None
            else "lexical-v1"
        ),
        "normalizer_db_path": (
            str(runner.retriever.normalizer.client.db_path)
            if runner.retriever.normalizer is not None
            else None
        ),
        "normalizer_sources": (
            list(runner.retriever.normalizer.client.source_vocabularies)
            if runner.retriever.normalizer is not None
            else []
        ),
        "case_count": len(case_list),
        "prediction_count": len(records),
        "argument_trace_count": argument_trace_count,
        "modes": [mode.value for mode in mode_list],
        "top_k": runner.top_k,
        "dataset_counts": dict(
            sorted(
                {
                    dataset: sum(
                        case.dataset == dataset for case in case_list
                    )
                    for dataset in {case.dataset for case in case_list}
                }.items()
            )
        ),
        "argumentation": {
            "central_scheme": ArgumentScheme.BEST_EXPLANATION.value,
            "reasoner_model_id": runner.reasoner_model.model_id,
            "verifier_model_id": runner.verifier_model.model_id,
            "resolver_id": "preference-grounded-bipolar-v1",
            "shared_agent_exchange_between_argument_modes": True,
        },
        "error_count": sum(record.error is not None for record in records),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return ExperimentRun(
        run_id=run_id,
        output_dir=run_dir,
        predictions_path=predictions_path,
        argument_traces_path=argument_traces_path,
        manifest_path=manifest_path,
        records=tuple(records),
    )
