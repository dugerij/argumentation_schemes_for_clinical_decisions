import json
from dataclasses import replace

import pytest

from clinical_cds.clinical_acceptability_judge import (
    judge_prediction, parse_clinical_judgment, summarize_clinical_judgments,
)
from clinical_cds.schema import DiagnosticGraph, ExperimentMode, PredictionRecord


class JudgeModel:
    model_id = "test:medical-family-judge"
    def __init__(self, same_family): self.same_family = same_family; self.prompts = []
    def complete(self, system_prompt, user_prompt, **kwargs):
        self.prompts.append((system_prompt, user_prompt, kwargs))
        return json.dumps({"relationship": (
                               "parent_child" if self.same_family else "different_disease"
                           ), "rationale": "Both labels identify the same disease process."})


class FlakyJudgeModel(JudgeModel):
    def complete(self, system_prompt, user_prompt, **kwargs):
        self.prompts.append((system_prompt, user_prompt, kwargs))
        if len(self.prompts) < 3:
            return "not-json"
        return json.dumps({"relationship": (
                               "parent_child" if self.same_family else "different_disease"
                           ), "rationale": "Same core disease identity."})


class LiteralControlCharacterJudgeModel(JudgeModel):
    def complete(self, system_prompt, user_prompt, **kwargs):
        self.prompts.append((system_prompt, user_prompt, kwargs))
        return ('{"relationship":"parent_child","rationale":'
                '"Same core disease\nidentity."}')


class InvalidJudgeModel(JudgeModel):
    def complete(self, system_prompt, user_prompt, **kwargs):
        self.prompts.append((system_prompt, user_prompt, kwargs))
        return "not-json"


class SequenceJudgeModel(JudgeModel):
    def complete(self, system_prompt, user_prompt, **kwargs):
        self.prompts.append((system_prompt, user_prompt, kwargs))
        evaluated = json.loads(user_prompt)["evaluated_diagnosis"]
        return json.dumps({"relationship": (
            "parent_child" if evaluated == "Aortic Dissection" else "different_disease"
        ), "rationale": "Structured family comparison."})


def record_for(case, predicted, *, abstained=False, error=None, metadata=None):
    return PredictionRecord(run_id="r", case_id=case.case_id, dataset=case.dataset,
        task=case.task, mode=ExperimentMode.GRAPH_RAG, model_id="producer",
        gold_label="Type A Aortic Dissection", predicted_label=predicted,
        reasoning="", citations=(), observations=(), abstained=abstained,
        latency_seconds=0, prompt_hash="", cache_hit=False,
        valid_evidence_ids=(), quality_flags=(), error=error,
        metadata=dict(metadata or {}))


def test_parent_and_child_same_family_pass(direct_root):
    from clinical_cds.direct import load_direct_dataset
    dataset = load_direct_dataset(direct_root); case = dataset.cases[0]
    model = JudgeModel(True)
    result = judge_prediction(record=record_for(case, "Aortic Dissection"),
        case=case, graphs=dataset.graphs, model=model)
    assert result.passed and result.same_disease_family
    submitted = json.loads(model.prompts[0][1])
    assert "mode" not in submitted and "model_id" not in submitted
    assert submitted["decision_rule"].startswith("PASS iff")
    system = model.prompts[0][0]
    assert "CORE\nDISEASE IDENTITY" in system
    assert "sibling diseases" in system
    assert "LOWEST meaningful\nCOMMON DIAGNOSTIC ANCESTOR" in system
    assert "sibling branches pass" in system
    assert "crisis, exacerbated, or decompensated" in system
    assert "named diagnostic syndrome" in system
    assert "conditions that can coexist" in system


def test_prompt_does_not_use_known_benchmark_diagnoses_as_examples():
    from clinical_cds.clinical_acceptability_judge import JUDGE_SYSTEM_PROMPT
    forbidden = (
        "pulmonary embolism", "aortic dissection", "acute coronary syndrome",
        "nstemi", "myocardial infarction", "heart failure", "pneumonia",
        "asthma", "chronic obstructive pulmonary disease",
        "upper gastrointestinal bleeding", "peptic ulcer disease",
        "hypertension", "hypertensive crisis", "stemi", "nste-acs",
    )
    prompt = JUDGE_SYSTEM_PROMPT.casefold()
    assert all(label not in prompt for label in forbidden)


def test_different_family_fails(direct_root):
    from clinical_cds.direct import load_direct_dataset
    dataset = load_direct_dataset(direct_root); case = dataset.cases[0]
    result = judge_prediction(record=record_for(case, "Heart Failure"), case=case,
        graphs=dataset.graphs, model=JudgeModel(False))
    assert not result.passed


def test_prediction_set_passes_when_non_primary_possibility_matches(direct_root):
    from clinical_cds.direct import load_direct_dataset
    dataset = load_direct_dataset(direct_root); case = dataset.cases[0]
    record = record_for(case, "Heart Failure", metadata={
        "possible_diagnoses": ["Heart Failure", "Aortic Dissection"]})
    model = SequenceJudgeModel(False)
    result = judge_prediction(record=record, case=case, graphs=dataset.graphs, model=model)
    assert result.passed
    assert result.decision_source == "prediction_set_family_match"


def test_abstention_fails_without_model_call(direct_root):
    from clinical_cds.direct import load_direct_dataset
    dataset = load_direct_dataset(direct_root); case = dataset.cases[0]
    model = JudgeModel(True)
    result = judge_prediction(record=record_for(case, "", abstained=True),
        case=case, graphs=dataset.graphs, model=model)
    assert not result.passed and result.included_in_accuracy and not model.prompts


def test_verified_insufficient_evidence_abstention_is_excluded(direct_root):
    from clinical_cds.direct import load_direct_dataset
    dataset = load_direct_dataset(direct_root); case = dataset.cases[0]
    model = JudgeModel(True)
    record = record_for(case, "", abstained=True,
        metadata={"abstention_category": "insufficient_evidence"})
    result = judge_prediction(record=record, case=case, graphs=dataset.graphs,
        model=model)
    assert not result.passed and not result.included_in_accuracy
    assert result.exclusion_reason == "insufficient_evidence_abstention"
    summary = summarize_clinical_judgments((record,), (result,))[0]
    assert summary["n"] == 1 and summary["evaluated_n"] == 0
    assert summary["family_accuracy"] is None
    assert summary["diagnostic_coverage"] == 0
    assert summary["accuracy_denominator_coverage"] == 0


def test_execution_failure_abstention_remains_a_scored_failure(direct_root):
    from clinical_cds.direct import load_direct_dataset
    dataset = load_direct_dataset(direct_root); case = dataset.cases[0]
    record = record_for(case, "", abstained=True, error="generation failed",
        metadata={"abstention_category": "execution_failure"})
    result = judge_prediction(record=record, case=case, graphs=dataset.graphs,
        model=JudgeModel(True))
    summary = summarize_clinical_judgments((record,), (result,))[0]
    assert result.included_in_accuracy and not result.passed
    assert summary["evaluated_n"] == 1 and summary["family_accuracy"] == 0
    assert summary["diagnostic_coverage"] == 0
    assert summary["accuracy_denominator_coverage"] == 1


def test_normalized_identical_label_passes_without_model_call(direct_root):
    from clinical_cds.direct import load_direct_dataset
    dataset = load_direct_dataset(direct_root); case = dataset.cases[0]
    record = replace(record_for(case, "Type-A aortic dissection"),
                     gold_label="Type A Aortic Dissection")
    model = JudgeModel(False)
    result = judge_prediction(record=record, case=case, graphs=dataset.graphs, model=model)
    assert result.passed and result.attempt_count == 0 and not model.prompts


def test_judge_retries_invalid_structured_output(direct_root):
    from clinical_cds.direct import load_direct_dataset
    dataset = load_direct_dataset(direct_root); case = dataset.cases[0]
    model = FlakyJudgeModel(True)
    result = judge_prediction(record=record_for(case, "Aortic Dissection"),
        case=case, graphs=dataset.graphs, model=model)
    assert result.passed and result.attempt_count == 3


def test_exhausted_judge_does_not_destroy_frozen_prediction(direct_root):
    from clinical_cds.direct import load_direct_dataset
    dataset = load_direct_dataset(direct_root); case = dataset.cases[0]
    record = record_for(case, "Unmapped diagnosis")
    result = judge_prediction(record=record, case=case, graphs=(),
        model=InvalidJudgeModel(False))
    summary = summarize_clinical_judgments((record,), (result,))[0]
    assert result.decision_source == "judge_unavailable"
    assert not result.included_in_accuracy
    assert summary["evaluation_failure_count"] == 1
    assert summary["evaluated_n"] == 0
    assert summary["diagnostic_answer_count"] == 1


def test_judge_accepts_literal_control_character_inside_structured_string(direct_root):
    from clinical_cds.direct import load_direct_dataset
    dataset = load_direct_dataset(direct_root); case = dataset.cases[0]
    result = judge_prediction(record=record_for(case, "Aortic Dissection"),
        case=case, graphs=dataset.graphs,
        model=LiteralControlCharacterJudgeModel(True))
    assert result.passed and result.attempt_count == 1


def test_parser_rejects_redundant_boolean_family_decision():
    with pytest.raises(ValueError, match="frozen schema"):
        parse_clinical_judgment({"relationship": "parent_child",
                                 "same_disease_family": "yes", "rationale": "x"})


def test_parser_derives_verdict_from_relationship():
    assert parse_clinical_judgment({
        "relationship": "sibling_branches", "rationale": "x"
    })[0]
    assert not parse_clinical_judgment({
        "relationship": "different_disease", "rationale": "x"
    })[0]


def test_summary_rejects_judgment_not_bound_to_frozen_prediction(direct_root):
    from clinical_cds.direct import load_direct_dataset
    dataset = load_direct_dataset(direct_root); case = dataset.cases[0]
    record = record_for(case, "Aortic Dissection")
    result = judge_prediction(record=record, case=case, graphs=dataset.graphs,
        model=JudgeModel(True))
    with pytest.raises(ValueError, match="frozen prediction"):
        summarize_clinical_judgments((replace(record, predicted_label="Other"),), (result,))


def test_identical_outputs_from_different_methods_do_not_collide(direct_root):
    from clinical_cds.direct import load_direct_dataset
    dataset = load_direct_dataset(direct_root); case = dataset.cases[0]
    graph = record_for(case, "Aortic Dissection")
    direct = replace(graph, mode=ExperimentMode.DIRECT)
    model = JudgeModel(True)
    results = tuple(judge_prediction(record=record, case=case,
        graphs=dataset.graphs, model=model) for record in (direct, graph))
    summaries = summarize_clinical_judgments((direct, graph), results)
    assert {item["mode"]: item["n"] for item in summaries} == {
        "direct": 1, "graph_rag": 1,
    }
    assert results[0].prediction_sha256 != results[1].prediction_sha256


def test_controlled_named_syndrome_sibling_branches_pass_without_model_call(
    direct_root,
):
    from clinical_cds.direct import load_direct_dataset
    dataset = load_direct_dataset(direct_root); case = dataset.cases[0]
    graph = DiagnosticGraph(
        graph_id="graph:epilepsy-syndrome", category="Epilepsy Syndrome",
        nodes=(), edges=(), diagnostic_paths={
            "p1": ("Suspected Epilepsy", "Focal Epilepsy"),
            "p2": ("Suspected Epilepsy", "Generalized Epilepsy"),
        },
    )
    record = replace(record_for(case, "Generalized Epilepsy"),
                     gold_label="Focal Epilepsy")
    model = JudgeModel(False)
    result = judge_prediction(
        record=record, case=case, graphs=(graph,), model=model
    )
    assert result.passed
    assert result.decision_source == "deterministic_controlled_graph_family"
    assert not model.prompts
