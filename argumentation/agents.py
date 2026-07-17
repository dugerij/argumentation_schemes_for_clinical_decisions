"""LLM-driven argumentation helpers for recommendation runs.

The aim here is simple:
- retrieve note evidence
- ask the generator to make claims tied to that evidence
- ask the verifier to challenge unsupported claims
- ask the reasoner to summarise the accepted position

The code stays close to that flow so it is easy to explain.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Any, Iterable, List, Optional

from dotenv import load_dotenv

from helpers.config import get_model_name
from helpers.jsonl import JsonlLogger
from helpers.ollama import ollama_chat
from helpers.paths import EVENT_LOG_PATH

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ArgMed")


@dataclass
class DialogueTurn:
    """Stores the interaction state for a single round of dialogue."""

    round_num: int
    generator_argument: str
    verifier_status: str
    verifier_cq: Optional[str] = None
    generator_payload: dict[str, Any] | None = None
    verifier_payload: dict[str, Any] | None = None


def _clean_json_content(content: str) -> str:
    """Strip common Markdown fences from model JSON output."""
    return content.replace("```json", "").replace("```", "").strip()


def _parse_json_payload(content: str) -> dict[str, Any]:
    """Parse a model response into a dictionary payload."""
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        payload = json.loads(_clean_json_content(content))
    return payload if isinstance(payload, dict) else {"value": payload}


def _as_text(value: Any) -> str | None:
    """Normalise a value into a single-line string."""
    if value is None:
        return None
    text = " ".join(str(value).split()).strip()
    return text or None


def _normalize_string_list(value: Any) -> list[str]:
    """Return a deduplicated list of plain strings."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        value = [value]
    output: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = _as_text(item)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(text)
    return output


def _normalize_evidence_ids(value: Any, allowed_ids: set[str]) -> list[str]:
    """Keep only valid evidence ids from model output."""
    raw_ids = _normalize_string_list(value)
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_id in raw_ids:
        evidence_id = raw_id.strip().upper().strip("[](),.;")
        if evidence_id not in allowed_ids or evidence_id in seen:
            continue
        seen.add(evidence_id)
        normalized.append(evidence_id)
    return normalized


def _normalize_argument_item(item: Any, allowed_ids: set[str]) -> dict[str, Any]:
    """Coerce one model-produced argument into a stable shape."""
    if not isinstance(item, dict):
        item = {"claim": item}

    claim = (
        _as_text(item.get("claim"))
        or _as_text(item.get("argument"))
        or _as_text(item.get("text"))
        or "Unspecified claim."
    )
    rationale = (
        _as_text(item.get("rationale"))
        or _as_text(item.get("why"))
        or _as_text(item.get("reason"))
    )
    evidence_ids = _normalize_evidence_ids(
        item.get("evidence_ids") or item.get("evidence") or item.get("citations"),
        allowed_ids,
    )
    return {
        "claim": claim,
        "rationale": rationale,
        "evidence_ids": evidence_ids,
    }


def _normalize_generator_payload(payload: dict[str, Any], allowed_ids: set[str]) -> dict[str, Any]:
    """Normalise generator output into the format used by the pipeline."""
    arguments = [
        _normalize_argument_item(item, allowed_ids)
        for item in payload.get("arguments", [])
    ]
    counterarguments = [
        _normalize_argument_item(item, allowed_ids)
        for item in payload.get("counterarguments", [])
    ]
    decision = (
        _as_text(payload.get("decision"))
        or _as_text(payload.get("recommendation"))
        or "Insufficient evidence for a grounded recommendation."
    )
    return {
        "decision": decision,
        "arguments": arguments,
        "counterarguments": counterarguments,
        "uncertainties": _normalize_string_list(payload.get("uncertainties")),
    }


def _normalize_verifier_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalise verifier output into the expected accept/reject schema."""
    status = _as_text(payload.get("status")) or "REJECT"
    status = status.upper()
    if status not in {"ACCEPT", "REJECT"}:
        status = "REJECT"
    return {
        "status": status,
        "critical_question": _as_text(payload.get("critical_question")),
        "notes": _normalize_string_list(payload.get("notes")),
    }


def _normalize_reasoner_payload(payload: dict[str, Any], allowed_ids: set[str]) -> dict[str, Any]:
    """Normalise the final reasoner summary."""
    accepted_arguments = [
        _normalize_argument_item(item, allowed_ids)
        for item in payload.get("accepted_arguments", [])
    ]
    rejected_arguments = [
        _normalize_argument_item(item, allowed_ids)
        for item in payload.get("rejected_arguments", [])
    ]
    decision = (
        _as_text(payload.get("decision"))
        or _as_text(payload.get("final_decision"))
        or "Insufficient evidence for a grounded recommendation."
    )
    summary = (
        _as_text(payload.get("summary"))
        or _as_text(payload.get("decision_rationale"))
    )
    return {
        "decision": decision,
        "accepted_arguments": accepted_arguments,
        "rejected_arguments": rejected_arguments,
        "open_questions": _normalize_string_list(
            payload.get("open_questions") or payload.get("uncertainties")
        ),
        "summary": summary,
    }


def _format_evidence_bundle(evidence_bundle: list[dict[str, Any]]) -> str:
    """Render retrieved evidence into a short prompt-friendly list."""
    if not evidence_bundle:
        return "No explicit evidence items were retrieved."

    lines: list[str] = []
    for item in evidence_bundle:
        evidence_id = item.get("evidence_id", "E?")
        source_name = item.get("source_name") or "unknown_source"
        section = item.get("section") or "unknown_section"
        score = item.get("score")
        score_text = f" score={score:.3f}" if isinstance(score, float) else ""
        lines.append(
            f"{evidence_id} | source={source_name} | section={section}{score_text} | {item.get('snippet', '')}"
        )
    return "\n".join(lines)


def _build_evidence_index(evidence_bundle: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index evidence items by evidence id for formatting."""
    return {
        str(item.get("evidence_id")): item
        for item in evidence_bundle
        if item.get("evidence_id")
    }


def _render_evidence_refs(evidence_ids: Iterable[str], evidence_index: dict[str, dict[str, Any]]) -> list[str]:
    """Render human-readable evidence references for one claim."""
    refs: list[str] = []
    for evidence_id in evidence_ids:
        item = evidence_index.get(evidence_id)
        if item is None:
            refs.append(f"[{evidence_id}]")
            continue
        source_name = item.get("source_name") or "unknown_source"
        section = item.get("section") or "unknown_section"
        snippet = item.get("snippet") or ""
        refs.append(f"[{evidence_id}] {source_name} / {section}: {snippet}")
    return refs


def render_generator_argument(
    payload: dict[str, Any],
    evidence_bundle: list[dict[str, Any]] | None = None,
) -> str:
    """Render generator output as readable text for logs and API output."""
    evidence_index = _build_evidence_index(evidence_bundle or [])
    lines = [f"Decision: {payload.get('decision', '')}"]

    arguments = payload.get("arguments", [])
    if arguments:
        lines.append("Supporting arguments:")
        for idx, argument in enumerate(arguments, start=1):
            lines.append(f"{idx}. {argument.get('claim')}")
            rationale = argument.get("rationale")
            if rationale:
                lines.append(f"   Why: {rationale}")
            refs = _render_evidence_refs(argument.get("evidence_ids", []), evidence_index)
            if refs:
                lines.append("   Evidence:")
                for ref in refs:
                    lines.append(f"   - {ref}")

    counterarguments = payload.get("counterarguments", [])
    if counterarguments:
        lines.append("Counterarguments:")
        for idx, argument in enumerate(counterarguments, start=1):
            lines.append(f"{idx}. {argument.get('claim')}")
            rationale = argument.get("rationale")
            if rationale:
                lines.append(f"   Why: {rationale}")
            refs = _render_evidence_refs(argument.get("evidence_ids", []), evidence_index)
            if refs:
                lines.append("   Evidence:")
                for ref in refs:
                    lines.append(f"   - {ref}")

    uncertainties = payload.get("uncertainties", [])
    if uncertainties:
        lines.append("Uncertainties:")
        for uncertainty in uncertainties:
            lines.append(f"- {uncertainty}")

    return "\n".join(lines).strip()


def render_final_assessment(
    payload: dict[str, Any],
    evidence_bundle: list[dict[str, Any]] | None = None,
) -> str:
    """Render the final reasoner output as a readable recommendation."""
    evidence_index = _build_evidence_index(evidence_bundle or [])
    lines = [f"Decision: {payload.get('decision', '')}"]

    accepted_arguments = payload.get("accepted_arguments", [])
    if accepted_arguments:
        lines.append("Accepted arguments:")
        for idx, argument in enumerate(accepted_arguments, start=1):
            lines.append(f"{idx}. {argument.get('claim')}")
            rationale = argument.get("rationale")
            if rationale:
                lines.append(f"   Why: {rationale}")
            refs = _render_evidence_refs(argument.get("evidence_ids", []), evidence_index)
            if refs:
                lines.append("   Evidence:")
                for ref in refs:
                    lines.append(f"   - {ref}")

    rejected_arguments = payload.get("rejected_arguments", [])
    if rejected_arguments:
        lines.append("Rejected or cautioned arguments:")
        for idx, argument in enumerate(rejected_arguments, start=1):
            lines.append(f"{idx}. {argument.get('claim')}")
            rationale = argument.get("rationale")
            if rationale:
                lines.append(f"   Why not accepted: {rationale}")
            refs = _render_evidence_refs(argument.get("evidence_ids", []), evidence_index)
            if refs:
                lines.append("   Evidence:")
                for ref in refs:
                    lines.append(f"   - {ref}")

    open_questions = payload.get("open_questions", [])
    if open_questions:
        lines.append("Open questions:")
        for question in open_questions:
            lines.append(f"- {question}")

    summary = payload.get("summary")
    if summary:
        lines.append(f"Summary: {summary}")

    return "\n".join(lines).strip()


class GeneratorAgent:
    """Ask the generation model for note-backed clinical claims."""

    def __init__(self, model: str):
        self.model = model
        logger.info("Initialized GeneratorAgent with model: %s", self.model)

    def generate(
        self,
        question: str,
        rag_context: str,
        evidence_bundle: list[dict[str, Any]],
        previous_critique: str | None = None,
    ) -> dict[str, Any]:
        """Generate a structured argument payload tied to evidence ids."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a medical reasoning generator. "
                    "Use only the provided evidence bundle. "
                    "Return valid JSON with keys: decision, arguments, counterarguments, uncertainties. "
                    "Each item in arguments or counterarguments must contain claim, rationale, and evidence_ids. "
                    "Every argument must cite at least one evidence id from the bundle, like E1 or E2. "
                    "If the evidence is insufficient, say so explicitly."
                ),
            }
        ]

        prompt = (
            f"Question:\n{question}\n\n"
            f"Retrieved context summary:\n{rag_context}\n\n"
            f"Evidence bundle:\n{_format_evidence_bundle(evidence_bundle)}\n"
        )
        if previous_critique:
            prompt += (
                "\nVerifier critique to address:\n"
                f"{previous_critique}\n"
                "Revise the arguments so every retained claim is grounded in the evidence bundle.\n"
            )

        messages.append({"role": "user", "content": prompt})

        logger.info("Generator is formulating an evidence-backed argument...")
        content = ollama_chat(
            model=self.model,
            messages=messages,
            format="json",
        )
        payload = _parse_json_payload(content)
        allowed_ids = {str(item.get("evidence_id")) for item in evidence_bundle if item.get("evidence_id")}
        return _normalize_generator_payload(payload, allowed_ids)


class VerifierAgent:
    """Ask the verifier model to reject unsupported or unsafe claims."""

    def __init__(self, model: str):
        self.model = model
        logger.info("Initialized VerifierAgent with model: %s", self.model)

    def verify(
        self,
        question: str,
        argument_payload: dict[str, Any],
        evidence_bundle: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Check whether the proposed claims are supported by the evidence bundle."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a medical critical verifier. Evaluate whether the proposed arguments are supported "
                    "by the provided evidence bundle and whether they contain contradictions or clinically material harms. "
                    "Reject the argument set if a material claim lacks evidence support or if the decision conflicts with the cited evidence. "
                    "Respond only in valid JSON with keys: status, critical_question, notes."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Candidate argument payload:\n{json.dumps(argument_payload, ensure_ascii=False, indent=2)}\n\n"
                    f"Evidence bundle:\n{_format_evidence_bundle(evidence_bundle)}"
                ),
            },
        ]

        logger.info("Verifier is evaluating the evidence-backed argument...")
        content = ollama_chat(
            model=self.model,
            messages=messages,
            format="json",
        )
        return _normalize_verifier_payload(_parse_json_payload(content))


class ReasonerAgent:
    """Ask the reasoner model for the final accepted position."""

    def __init__(self, model: str):
        self.model = model
        logger.info("Initialized ReasonerAgent with model: %s", self.model)

    def reason(
        self,
        dialogue_history: List[DialogueTurn],
        evidence_bundle: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Summarise the dialogue into a final structured decision."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the Argumentation Framework Reasoner. Review the dialogue between the Generator and Verifier. "
                    "Produce a final evidence-backed recommendation. "
                    "Return only valid JSON with keys: decision, accepted_arguments, rejected_arguments, open_questions, summary. "
                    "Each accepted or rejected argument must contain claim, rationale, and evidence_ids."
                ),
            }
        ]

        history_payload = [asdict(turn) for turn in dialogue_history]
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Dialogue history:\n{json.dumps(history_payload, ensure_ascii=False, indent=2)}\n\n"
                    f"Evidence bundle:\n{_format_evidence_bundle(evidence_bundle)}"
                ),
            }
        )

        logger.info("Reasoner is assessing the final framework...")
        content = ollama_chat(
            model=self.model,
            messages=messages,
            format="json",
        )
        allowed_ids = {str(item.get("evidence_id")) for item in evidence_bundle if item.get("evidence_id")}
        return _normalize_reasoner_payload(_parse_json_payload(content), allowed_ids)


class ArgumentInteraction:
    """Run generator, verifier, and reasoner in sequence for one question."""

    def __init__(
        self,
        question: str,
        rag_context: str,
        max_rounds: int,
        evidence_bundle: list[dict[str, Any]] | None = None,
        event_logger: JsonlLogger | None = None,
    ):
        self.question = question
        self.rag_context = rag_context
        self.max_rounds = max_rounds
        self.evidence_bundle = evidence_bundle or []
        self.dialogue_history: List[DialogueTurn] = []
        self.event_logger = event_logger or JsonlLogger(EVENT_LOG_PATH, run_id="argument_interaction")
        self.final_assessment_payload: dict[str, Any] | None = None
        self.formatted_final_assessment: str | None = None

        self.generator = GeneratorAgent(get_model_name("GENERATOR"))
        self.verifier = VerifierAgent(get_model_name("VERIFIER"))
        self.reasoner = ReasonerAgent(get_model_name("REASONER"))

    def run(self) -> str:
        """Execute the argumentation loop and return the final rendered output."""
        logger.info("Starting ArgMed Pipeline Interaction")
        self.event_logger.event(
            "argument_interaction",
            "started",
            max_rounds=self.max_rounds,
            question_preview=self.question[:500],
            context_chars=len(self.rag_context),
            evidence_count=len(self.evidence_bundle),
        )
        previous_critique = None

        for round_num in range(1, self.max_rounds + 1):
            logger.info("--- Beginning Round %s ---", round_num)
            self.event_logger.event("argument_round", "started", round_num=round_num)

            with self.event_logger.timed("generator", round_num=round_num):
                argument_payload = self.generator.generate(
                    self.question,
                    self.rag_context,
                    self.evidence_bundle,
                    previous_critique,
                )
            argument_text = render_generator_argument(argument_payload, self.evidence_bundle)
            logger.info("GENERATOR OUTPUT:\n%s", argument_text)
            self.event_logger.event(
                "generator",
                "output",
                round_num=round_num,
                argument_preview=argument_text[:2000],
                argument_chars=len(argument_text),
                argument_payload=argument_payload,
            )

            with self.event_logger.timed("verifier", round_num=round_num):
                verification = self.verifier.verify(
                    self.question,
                    argument_payload,
                    self.evidence_bundle,
                )
            status = verification.get("status", "REJECT").upper()
            cq = verification.get("critical_question")
            logger.info("VERIFIER STATUS: %s | CQ: %s", status, cq)
            self.event_logger.event(
                "verifier",
                "output",
                round_num=round_num,
                verifier_status=status,
                critical_question=cq,
                raw_verification=verification,
            )

            turn = DialogueTurn(
                round_num=round_num,
                generator_argument=argument_text,
                verifier_status=status,
                verifier_cq=cq,
                generator_payload=argument_payload,
                verifier_payload=verification,
            )
            self.dialogue_history.append(turn)

            if status == "ACCEPT":
                logger.info("Argument accepted. Ending dialogue loop.")
                self.event_logger.event("argument_round", "completed", round_num=round_num, accepted=True)
                break

            previous_critique = cq
            self.event_logger.event("argument_round", "completed", round_num=round_num, accepted=False)
            if round_num == self.max_rounds:
                logger.warning("Max rounds reached with unresolved conflicts.")
                self.event_logger.event("argument_interaction", "max_rounds_reached", max_rounds=self.max_rounds)

        logger.info("--- Handing over to Reasoner ---")
        with self.event_logger.timed("reasoner", turns=len(self.dialogue_history)):
            final_assessment_payload = self.reasoner.reason(self.dialogue_history, self.evidence_bundle)
        final_assessment = render_final_assessment(final_assessment_payload, self.evidence_bundle)
        self.final_assessment_payload = final_assessment_payload
        self.formatted_final_assessment = final_assessment
        logger.info("FINAL ASSESSMENT:\n%s", final_assessment)
        self.event_logger.event(
            "argument_interaction",
            "completed",
            turns=len(self.dialogue_history),
            final_assessment_preview=final_assessment[:2000],
            final_assessment_chars=len(final_assessment),
            final_assessment_payload=final_assessment_payload,
        )

        return final_assessment

    def dump_history(self) -> str:
        """Utility to export the dialogue history as JSON."""

        return json.dumps([asdict(turn) for turn in self.dialogue_history], indent=2)


if __name__ == "__main__":
    MAX_ROUNDS = int(os.getenv("MAX_ROUNDS", 3))

    medqa_question = (
        "An 18-year-old woman presents with recurrent headaches. The pain is usually unilateral, "
        "pulsatile in character, exacerbated by light and noise, and usually lasts for a few hours "
        "to a full day. The pain is sometimes triggered by eating chocolates. These headaches disturb "
        "her daily routine activities. The physical examination was within normal limits. She also has "
        "essential tremors. Which drug is suitable in her case for the prevention of headaches?"
    )

    mock_rag_context = (
        "Migraine presentation with photophobia and phonophobia. "
        "Prophylactic options include propranolol and topiramate. "
        "Essential tremor also responds to propranolol."
    )
    mock_evidence_bundle = [
        {
            "evidence_id": "E1",
            "source_name": "mock_note.txt",
            "section": "history_of_present_illness",
            "snippet": "Migraine presentation: unilateral, pulsatile, photophobia, phonophobia.",
        },
        {
            "evidence_id": "E2",
            "source_name": "mock_note.txt",
            "section": "brief_hospital_course",
            "snippet": "Essential tremor is treated with propranolol.",
        },
    ]

    interaction = ArgumentInteraction(
        question=medqa_question,
        rag_context=mock_rag_context,
        max_rounds=MAX_ROUNDS,
        evidence_bundle=mock_evidence_bundle,
    )

    final_output = interaction.run()

    print("\n" + "=" * 50)
    print("PIPELINE COMPLETE. FINAL REASONER OUTPUT:")
    print("=" * 50)
    print(final_output)
