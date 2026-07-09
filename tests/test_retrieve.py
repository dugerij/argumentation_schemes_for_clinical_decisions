import asyncio
from types import SimpleNamespace

from retrieval.query import query_index_context, query_index_with_diagnostics


class _Response:
    source_nodes = []

    def __str__(self) -> str:
        return "answer"


class _QueryEngine:
    def query(self, query: str) -> _Response:
        assert query == "question"
        return _Response()


class _Index:
    def __init__(self) -> None:
        self._llm = object()
        self.received_llm = None

    def as_query_engine(self, *, similarity_top_k: int, llm):
        assert similarity_top_k == 5
        self.received_llm = llm
        return _QueryEngine()


class _Node:
    def __init__(self, text: str) -> None:
        self.text = text
        self.metadata = {}


class _NodeWithScore:
    def __init__(self, text: str, score: float) -> None:
        self.node = _Node(text)
        self.score = score


class _DiagnosticResponse:
    def __init__(self) -> None:
        self.source_nodes = [
            _NodeWithScore("atrial fibrillation hypertension", 0.9),
            _NodeWithScore("hypertension aspirin", 0.7),
        ]

    def __str__(self) -> str:
        return "diagnostic answer"


class _DiagnosticQueryEngine:
    def query(self, query: str) -> _DiagnosticResponse:
        assert query == "atrial fibrillation treatment"
        return _DiagnosticResponse()


class _DiagnosticIndex(_Index):
    def as_query_engine(self, *, similarity_top_k: int, llm):
        assert similarity_top_k == 5
        self.received_llm = llm
        return _DiagnosticQueryEngine()


class _FakeEmbedModel:
    mapping = {
        "atrial fibrillation treatment": [1.0, 1.0, 0.0],
        "Text: atrial fibrillation hypertension": [1.0, 1.0, 0.0],
        "Text: hypertension aspirin": [1.0, 0.0, 0.0],
    }

    def get_text_embedding(self, text: str):
        return self.mapping[text]


class _FakeCompletion:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> _FakeCompletion:
        self.prompts.append(prompt)
        return _FakeCompletion("Use ACE inhibitor alternatives cautiously given CKD risk.")


class _ImplicitNode:
    def __init__(self, *, text: str, embedding: list[float], source_name: str) -> None:
        self.label = "text_chunk"
        self.text = text
        self.embedding = embedding
        self.metadata = {"source_name": source_name}


class _ImplicitIndex:
    def __init__(self) -> None:
        self._index_mode = "implicit"
        self._llm = _FakeLLM()
        self._embed_model = _FakeEmbedModel()
        self.property_graph_store = SimpleNamespace(
            graph=SimpleNamespace(
                nodes={
                    "a": _ImplicitNode(
                        text="Patient has chronic kidney disease stage IV with hypertension and recent dyspnea.",
                        embedding=[1.0, 1.0, 0.0],
                        source_name="note-a.txt",
                    ),
                    "b": _ImplicitNode(
                        text="Unrelated postoperative vascular surgery recovery note.",
                        embedding=[0.0, 0.0, 1.0],
                        source_name="note-b.txt",
                    ),
                }
            )
        )


def test_query_reuses_index_llm():
    index = _Index()

    response, context = asyncio.run(query_index_context(index, "question"))

    assert index.received_llm is index._llm
    assert response == "answer"
    assert context == "answer"


def test_query_returns_retrieval_diagnostics():
    index = _DiagnosticIndex()

    response, context, diagnostics = asyncio.run(
        query_index_with_diagnostics(
            index,
            "atrial fibrillation treatment",
            embed_model=_FakeEmbedModel(),
        )
    )

    assert response == "diagnostic answer"
    assert "atrial fibrillation hypertension" in context
    assert diagnostics["top_retrieval_score"] == 0.9
    assert diagnostics["mean_retrieval_score"] == 0.8
    assert round(diagnostics["retrieval_score_margin"], 4) == 0.2
    assert round(diagnostics["top_cosine_similarity"], 4) == 1.0
    assert diagnostics["mean_cosine_similarity"] is not None


def test_query_uses_direct_chunk_retrieval_for_implicit_indexes():
    index = _ImplicitIndex()

    response, context, diagnostics = asyncio.run(
        query_index_with_diagnostics(
            index,
            "atrial fibrillation treatment",
            embed_model=_FakeEmbedModel(),
        )
    )

    assert "CKD risk" in response
    assert "chronic kidney disease stage IV" in context
    assert "PREVIOUS" not in context
    assert diagnostics["source_count"] == 2
    assert diagnostics["top_retrieval_score"] is not None
