import sys
from types import SimpleNamespace
from unittest.mock import patch

from rag.vllm_offline import VLLMOfflineEmbedding


class _OldVLLM:
    init_kwargs = None

    def __init__(self, **kwargs):
        type(self).init_kwargs = kwargs

    def encode(self, texts):
        return [SimpleNamespace(outputs=SimpleNamespace(embedding=[float(len(text))])) for text in texts]


class _NewVLLM:
    init_kwargs = None

    def __init__(self, **kwargs):
        type(self).init_kwargs = kwargs

    def embed(self, texts):
        return [SimpleNamespace(outputs=SimpleNamespace(embedding=[float(len(text))])) for text in texts]


def test_old_vllm_selects_transformers_fallback():
    embedding = VLLMOfflineEmbedding(model_name="BAAI/bge-small-en-v1.5")
    with (
        patch("rag.vllm_offline.version", return_value="0.6.1.post1"),
        patch.object(embedding, "_ensure_transformer_loaded") as ensure_transformer,
    ):
        embedding._ensure_loaded()

    assert embedding._use_transformers is True
    ensure_transformer.assert_called_once_with()


def test_new_vllm_uses_embed_task():
    with (
        patch.dict(sys.modules, {"vllm": SimpleNamespace(LLM=_NewVLLM)}),
        patch("rag.vllm_offline.version", return_value="0.10.0"),
    ):
        embedding = VLLMOfflineEmbedding(model_name="BAAI/bge-small-en-v1.5")
        assert embedding._get_text_embeddings(["abcd"]) == [[4.0]]

    assert _NewVLLM.init_kwargs["task"] == "embed"
