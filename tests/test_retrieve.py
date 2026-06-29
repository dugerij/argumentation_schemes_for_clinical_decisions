import asyncio

from rag.retrieve import query_index_context


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


def test_query_reuses_index_llm():
    index = _Index()

    response, context = asyncio.run(query_index_context(index, "question"))

    assert index.received_llm is index._llm
    assert response == "answer"
    assert context == "answer"
