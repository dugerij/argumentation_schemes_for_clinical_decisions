import asyncio

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
