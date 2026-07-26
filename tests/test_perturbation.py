from clinical_cds.direct import load_direct_dataset
from clinical_cds.perturbation import (
    build_section_removal_pairs,
    evaluate_perturbations,
)
from clinical_cds.schema import (
    ExperimentMode,
    PredictionRecord,
)


def _record(case, citations):
    return PredictionRecord(
        run_id="run",
        case_id=case.case_id,
        dataset=case.dataset,
        task=case.task,
        mode=ExperimentMode.SYMBOLIC_ARGUMENT,
        model_id="test:model",
        gold_label=case.gold_label,
        predicted_label=case.gold_label,
        reasoning="",
        citations=tuple(citations),
        observations=(),
        abstained=False,
        latency_seconds=0.1,
        prompt_hash="hash",
        cache_hit=False,
        valid_evidence_ids=(),
    )


def test_section_removal_detects_stale_citation(direct_root):
    case = load_direct_dataset(direct_root).cases[0]
    pair = build_section_removal_pairs((case,))[0]
    records = (
        _record(pair.base_case, (pair.removed_evidence_id,)),
        _record(pair.perturbed_case, (pair.removed_evidence_id,)),
    )

    metrics = evaluate_perturbations((pair,), records)

    assert len(metrics) == 1
    assert metrics[0].base_cited_removed_section == 1.0
    assert metrics[0].stale_removed_section_citation == 1.0
    assert pair.removed_section not in pair.perturbed_case.sections
