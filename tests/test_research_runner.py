import json

from clinical_cds.direct import load_direct_dataset
from clinical_cds.evaluation import (
    evaluate_predictions,
    paired_comparisons,
    write_evaluation_artifacts,
)
from clinical_cds.retrieval import KnowledgeRetriever
from clinical_cds.runner import ExperimentRunner, run_experiment
from clinical_cds.schema import ExperimentMode
from clinical_cds.trace_visualization import (
    load_argument_trace,
    plot_argument_trace,
)


class FixedDiagnosticModel:
    model_id = "test:fixed"

    def __init__(self):
        self.call_count = 0

    def complete(
        self,
        system_prompt,
        user_prompt,
        *,
        output_schema=None,
    ):
        self.call_count += 1
        if "diagnostic reasoner agent" in system_prompt:
            return json.dumps(
                {
                    "candidates": [
                        {
                            "diagnosis": "Hypertension",
                            "arguments": [
                                {
                                    "scheme": "argument_from_diagnostic_criterion",
                                    "premise": "Repeated blood pressure satisfies the criterion.",
                                    "evidence_ids": ["S-PE", "K1"],
                                }
                            ],
                        },
                        {
                            "diagnosis": "Suspected Hypertension",
                            "arguments": [
                                {
                                    "scheme": "argument_from_risk_factor",
                                    "premise": "Family history increases prior plausibility.",
                                    "evidence_ids": ["S-FH", "K2"],
                                }
                            ],
                        },
                    ],
                    "preferred_diagnosis": "Suspected Hypertension",
                    "abstain": False,
                }
            )
        if "independent diagnostic verifier agent" in system_prompt:
            assert "A1" in user_prompt
            assert "A2" in user_prompt
            return json.dumps(
                {
                    "reviews": [
                        {
                            "argument_id": "A1",
                            "verdict": "supported",
                            "failed_critical_questions": [],
                            "explanation": "The observed result satisfies the criterion.",
                            "evidence_ids": ["S-PE", "K1"],
                        },
                        {
                            "argument_id": "A2",
                            "verdict": "supported",
                            "failed_critical_questions": [],
                            "explanation": "The history supports prior plausibility only.",
                            "evidence_ids": ["S-FH", "K2"],
                        },
                    ],
                    "counterarguments": [],
                    "abstain": False,
                }
            )

        citation = "K1" if "K1" in user_prompt else "S-PE"
        return json.dumps(
            {
                "answer": "Hypertension",
                "reasoning": "Repeated elevation supports the diagnosis.",
                "citations": ["S-PE", citation],
                "observations": [
                    {
                        "text": "Blood pressure is 170/100 mmHg.",
                        "source_id": "S-PE",
                    }
                ],
                "abstain": False,
            }
        )


class CountingRetriever:
    normalizer = None

    def __init__(self, delegate):
        self.delegate = delegate
        self.call_count = 0

    @property
    def retriever_id(self):
        return self.delegate.retriever_id

    def retrieve(self, case, *, top_k):
        self.call_count += 1
        return self.delegate.retrieve(case, top_k=top_k)


def test_runner_executes_all_ablations_and_shares_agent_exchange(
    tmp_path,
    direct_root,
    capsys,
):
    dataset = load_direct_dataset(direct_root)
    model = FixedDiagnosticModel()
    retriever = CountingRetriever(KnowledgeRetriever(dataset.graphs))
    runner = ExperimentRunner(
        model=model,
        retriever=retriever,
        cache_dir=tmp_path / "cache",
        top_k=3,
    )
    run = run_experiment(
        cases=dataset.cases,
        modes=tuple(ExperimentMode),
        runner=runner,
        output_dir=tmp_path / "runs",
        run_name="first",
        show_progress=True,
    )
    progress_output = capsys.readouterr().err

    assert len(run.records) == 5
    assert model.call_count == 5
    assert retriever.call_count == 1
    assert run.predictions_path.exists()
    assert run.argument_traces_path.exists()
    assert run.manifest_path.exists()
    assert "5/5" in progress_output
    assert "reasoner" in progress_output
    assert "verifier" in progress_output
    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    assert manifest["prompt_version"] == "diagnostic-symbolic-v2"
    assert set(manifest["model_configurations"]) == {
        "standard",
        "reasoner",
        "verifier",
    }
    trace = json.loads(
        run.argument_traces_path.read_text(encoding="utf-8").splitlines()[0]
    )
    assert trace["argument_graph"]["nodes"]
    assert trace["symbolic_resolution"]["resolver_id"] == (
        "preference-grounded-bipolar-v1"
    )
    assert "gold_label" not in trace
    selected_trace = load_argument_trace(
        run.argument_traces_path,
        case_id=dataset.cases[0].case_id,
    )
    trace_plot = plot_argument_trace(
        selected_trace,
        tmp_path / "argument_trace.png",
    )
    assert trace_plot.is_file()

    by_mode = {record.mode: record for record in run.records}
    assert (
        by_mode[ExperimentMode.STRUCTURED_ARGUMENT].predicted_label
        == "Suspected Hypertension"
    )
    assert (
        by_mode[ExperimentMode.SYMBOLIC_ARGUMENT].predicted_label
        == "Hypertension"
    )
    assert by_mode[
        ExperimentMode.SYMBOLIC_ARGUMENT
    ].metadata["argument_resolution_changed"] is True

    cached = runner.predict(
        dataset.cases[0],
        ExperimentMode.DIRECT,
        run_id="second",
    )
    assert cached.cache_hit is True
    assert model.call_count == 5
    assert retriever.call_count == 1


def test_deterministic_metrics_include_argument_quality(
    tmp_path,
    direct_root,
):
    dataset = load_direct_dataset(direct_root)
    runner = ExperimentRunner(
        model=FixedDiagnosticModel(),
        retriever=KnowledgeRetriever(dataset.graphs),
        cache_dir=tmp_path / "cache",
        top_k=3,
    )
    run = run_experiment(
        cases=dataset.cases,
        modes=tuple(ExperimentMode),
        runner=runner,
        output_dir=tmp_path / "runs",
        run_name="metrics",
    )

    metrics = evaluate_predictions(run.records, dataset.cases, dataset.graphs)
    comparisons = paired_comparisons(metrics)
    by_mode = {metric.mode: metric for metric in metrics}

    assert by_mode["structured_argument"].exact_match == 0.0
    assert by_mode["symbolic_argument"].exact_match == 1.0
    assert by_mode["symbolic_argument"].argument_schema_validity == 1.0
    assert by_mode["symbolic_argument"].argument_evidence_validity == 1.0
    assert by_mode["symbolic_argument"].verifier_review_coverage == 1.0
    assert by_mode["symbolic_argument"].symbolic_trace_fidelity == 1.0
    assert len(comparisons) == 4

    artifacts = write_evaluation_artifacts(
        records=run.records,
        cases=dataset.cases,
        graphs=dataset.graphs,
        output_dir=run.output_dir / "evaluation",
    )
    assert artifacts["accuracy_progression_plot"].is_file()
    assert artifacts["argument_quality_plot"].is_file()
    summary_text = artifacts["mode_summary"].read_text(encoding="utf-8")
    assert "argument_schema_validity" in summary_text
    assert "symbolic_trace_fidelity" in summary_text
