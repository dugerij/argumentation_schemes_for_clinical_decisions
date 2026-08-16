import importlib
import json
import sys
from pathlib import Path


def test_gpu_runtime_uses_pinned_qwen_judge_and_never_submits_jobs() -> None:
    source = Path("hf_job/run.py").read_text(encoding="utf-8").casefold()

    assert "medgemma-27b-text-it" not in source  # model identity comes from one config
    assert "qwen/qwen3-30b-a3b-instruct-2507-fp8" in source
    assert "reasoner_model=model" in source
    assert "model=judge_model" in source
    assert "run_job(" not in source
    assert "jobs run" not in source
    assert "release_certificate" not in source


def test_gpu_runtime_uses_the_canonical_current_argument_method() -> None:
    source = Path("hf_job/run.py").read_text(encoding="utf-8")

    assert "from clinical_cds.method_contract import CURRENT_ARGUMENT_METHOD" in source
    assert "argument_method=CURRENT_ARGUMENT_METHOD" in source


def test_paid_runtime_runs_blinded_family_judge_after_predictions_freeze() -> None:
    source = Path("hf_job/run.py").read_text(encoding="utf-8")

    experiment = source.index("comparison = run_experiment(")
    judging = source.index("judge_results = write_clinical_judgments(")
    assert judging > experiment
    assert '"clinical_family_judgments_sha256"' in source
    assert '"clinical_family_summary_sha256"' in source


def test_evaluation_resume_runs_before_retrieval_or_medgemma_startup() -> None:
    source = Path("hf_job/run.py").read_text(encoding="utf-8")

    branch = source.index('if RUN_PHASE == "evaluation"')
    preflight = source.index("argument_generation_preflight =", branch)
    assert branch < preflight
    resume_body = source[source.index("def _run_evaluation_resume("):source.index("def main()")]
    assert "write_clinical_judgments(" in resume_body
    assert "run_experiment(" not in resume_body
    assert '"medgemma_invocation_count": 0' in resume_body
    assert '"direct", "flat_rag", "graph_rag", "evidence_grounded_argumentation"' in resume_body


def test_evaluation_is_an_allowlisted_runtime_phase(monkeypatch) -> None:
    monkeypatch.setenv("RUN_ID", "evaluation-phase-test")
    monkeypatch.setenv("RUN_SCOPE", "development")
    monkeypatch.setenv("RUN_PHASE", "evaluation")
    sys.modules.pop("hf_job.config", None)
    config = importlib.import_module("hf_job.config")
    assert config.RUN_PHASE == "evaluation"


def test_both_generation_models_have_final_prompt_boundary_audits() -> None:
    source = Path("hf_job/run.py").read_text(encoding="utf-8")

    assert 'JUDGE_URL = "http://127.0.0.1:8005/v1"' in source
    assert 'JUDGE_UPSTREAM_URL = "http://127.0.0.1:8004/v1"' in source
    assert '"exact-qwen-chat-template-v1"' in source
    assert '"final_medgemma_prompt_boundary"' in source
    assert '"final_qwen_prompt_boundary"' in source


def test_graphrag_medgemma_prompt_audit_executes_its_validation_body(
    monkeypatch, tmp_path,
) -> None:
    monkeypatch.setenv("RUN_ID", "prompt-audit-regression")
    monkeypatch.setenv("RUN_SCOPE", "development")
    monkeypatch.setenv("RUN_PHASE", "comparison")
    sys.modules.pop("hf_job.run", None)
    sys.modules.pop("hf_job.config", None)
    runtime = importlib.import_module("hf_job.run")
    from graphrag_runtime.audit import GRAPHRAG_CANDIDATE_CHOICE_CONTRACT_ID

    path = tmp_path / "prompt.jsonl"
    record = {
        "rendered_input_tokens": 100,
        "requested_completion_tokens": 100,
        "safety_margin_tokens": 100,
        "max_model_len": 1000,
        "messages_sha256": "a" * 64,
        "rendered_token_ids_sha256": "b" * 64,
        "structured_output_contract_id": GRAPHRAG_CANDIDATE_CHOICE_CONTRACT_ID,
        "structured_output_injected": True,
        "response_schema_sha256": "c" * 64,
        "candidate_choice_count": 2,
        "candidate_source_pair_count": 2,
    }
    path.write_text(json.dumps(record) + "\n")

    result = runtime.audit_medgemma_prompt_boundary(path)

    assert result["status"] == "validated"
    assert result["request_count"] == 1


def test_paid_runtime_preflights_retry_budget_and_structured_output_bounds(
    monkeypatch,
) -> None:
    monkeypatch.setenv("RUN_ID", "argument-generation-preflight")
    monkeypatch.setenv("RUN_SCOPE", "development")
    monkeypatch.setenv("RUN_PHASE", "comparison")
    sys.modules.pop("hf_job.run", None)
    sys.modules.pop("hf_job.config", None)
    runtime = importlib.import_module("hf_job.run")

    result = runtime.validate_argument_generation_contract()

    assert result["status"] == "passed"
    assert result["maximum_completion_tokens"] == 3072
    assert max(result["stage_completion_tokens"].values()) <= 3072
    assert result["vllm_compatible_arrays_bounded"] is True


def test_gpu_runtime_is_supplied_by_the_pinned_image_not_installed_at_startup() -> None:
    source = Path("hf_job/run.py").read_text(encoding="utf-8")
    runbook = Path("RUNNING.md").read_text(encoding="utf-8")

    assert '"pip", "install", f"vllm==' not in source
    assert '"--system-site-packages"' in source
    assert "vllm/vllm-openai:v0.18.1" in runbook
    assert "pytorch/pytorch:2.6.0" not in runbook
    assert "python3 /workspace/project/hf_job/run.py" in runbook
    assert "python_executable = Path(sys.executable).resolve()" in source
    assert 'vllm_executable.parent / "python"' not in source


def test_readme_exposes_direct_hugging_face_commands() -> None:
    readme = Path("RUNNING.md").read_text(encoding="utf-8")

    assert "hf buckets sync" in readme
    assert "hf jobs run --detach" in readme
    assert "python -m hf_job.prepare" in readme
    assert "Choose a new `RUN_ID` for every job" in readme


def test_repository_readme_links_to_running_guide() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "[RUNNING.md](RUNNING.md)" in readme
    assert "Evidence-Grounded Clinical Argumentation" in readme


def test_documented_run_ids_are_lowercase() -> None:
    readme = Path("RUNNING.md").read_text(encoding="utf-8")

    assert "+%Y%m%dt%H%M%Sz" in readme
    assert "+%Y%m%dT%H%M%SZ" not in readme


def test_historical_and_generated_material_is_ignored() -> None:
    ignore = Path(".gitignore").read_text(encoding="utf-8")

    for entry in ("output/", "work/", ".local-history/", ".hf-runs/"):
        assert entry in ignore


def test_bounded_development_selection_is_reproducible_and_case_id_only(
    monkeypatch,
) -> None:
    monkeypatch.setenv("RUN_ID", "bounded-selection-test")
    monkeypatch.setenv("RUN_SCOPE", "development")
    monkeypatch.setenv("RUN_PHASE", "comparison")
    sys.modules.pop("hf_job.prepare", None)
    sys.modules.pop("hf_job.config", None)
    prepare = importlib.import_module("hf_job.prepare")
    records = [
        {"case_id": f"case-{index}", "gold_label": f"label-{index}"}
        for index in range(12)
    ]
    relabelled = [
        {**record, "gold_label": "changed"}
        for record in records
    ]

    first = prepare.select_bounded_records(records, limit=5, seed="fixed-v1")
    second = prepare.select_bounded_records(relabelled, limit=5, seed="fixed-v1")

    assert [item["case_id"] for item in first] == [
        item["case_id"] for item in second
    ]
    assert len(first) == 5
    assert len({item["case_id"] for item in first}) == 5


def test_runbook_documents_bounded_development_without_full_88_case_cost() -> None:
    readme = Path("RUNNING.md").read_text(encoding="utf-8")

    assert "RUN_CASE_LIMIT=12" in readme
    assert "RUN_SAMPLE_SEED=bounded-development-v1" in readme
    assert "for a bounded check" in readme


def test_runtime_accepts_cryptographically_bound_sample_in_retrieval_phase() -> None:
    source = Path("hf_job/run.py").read_text(encoding="utf-8")

    bounded_guard = source.split("if sample and (", 1)[1].split("):", 1)[0]
    assert 'RUN_PHASE != "comparison"' not in bounded_guard
    assert 'RUN_SCOPE != "development"' in bounded_guard
    assert 'sample.get("method") != "sha256-seeded-case-id-v1"' in bounded_guard
    assert 'sample.get("selection_uses") != "case_id_only"' in bounded_guard
