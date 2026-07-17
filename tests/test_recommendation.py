import asyncio
from unittest.mock import patch

from api.recommendation import RecommendationRequest, _has_graph, generate_recommendation


class _DummyTimed:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _DummyLogger:
    def __init__(self, *_args, **_kwargs):
        self.run_id = "run-1"

    def event(self, *_args, **_kwargs):
        return None

    def timed(self, *_args, **_kwargs):
        return _DummyTimed()


class _DummyInteraction:
    last_init = None

    def __init__(self, **kwargs):
        type(self).last_init = kwargs
        self.dialogue_history = []

    def run(self) -> str:
        return "Reasoning\nAnswer: pneumonia"


def test_has_graph_detects_materialized_case_graph(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "cds_case_graph.pkl.gz").write_bytes(b"graph")

    assert _has_graph(output_dir) is True


def test_generate_recommendation_uses_case_graph_and_passes_evidence_bundle(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "cds_case_graph.pkl.gz").write_bytes(b"graph")

    query = type(
        "Query",
        (),
        {
            "question": "Question text",
            "rag_context": "graph context",
            "evidence_bundle": [{"evidence_id": "E1", "snippet": "target case"}],
            "expected_answer": "pneumonia",
        },
    )()

    request = RecommendationRequest(case_id=101, task="diagnosis", dry_run=False)

    with patch("api.recommendation.load_dotenv"), patch("api.recommendation.new_run_id", return_value="run-1"), patch(
        "api.recommendation.JsonlLogger", _DummyLogger
    ), patch(
        "api.recommendation.CdsMaterializedGraphStore.from_persist_dir", return_value=object()
    ), patch(
        "api.recommendation.query_cds_evidence_bundle", return_value=query
    ), patch(
        "api.recommendation.ArgumentInteraction", _DummyInteraction
    ), patch(
        "api.recommendation.write_eval_record"
    ), patch.dict(
        "os.environ",
        {"OUTPUT_BASE_DIR": str(output_dir)},
        clear=False,
    ):
        response = asyncio.run(generate_recommendation(request))

    assert _DummyInteraction.last_init is not None
    assert _DummyInteraction.last_init["question"] == "Question text"
    assert _DummyInteraction.last_init["evidence_bundle"] == query.evidence_bundle
    assert response.rag_context == "graph context"
    assert response.retrieval_backend == "materialized_case_graph"
    assert response.final_recommendation == "Reasoning\nAnswer: pneumonia"
