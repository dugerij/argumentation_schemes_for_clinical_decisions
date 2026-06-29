from __future__ import annotations

import os
from collections.abc import Generator
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from llama_index.core.base.embeddings.base import Embedding
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import CompletionResponse, CustomLLM, LLMMetadata
from pydantic import Field, PrivateAttr


class VLLMOfflineLLM(CustomLLM):
    model_name: str = Field(default="Qwen/Qwen2.5-7B-Instruct")
    tensor_parallel_size: int = Field(default=1)
    gpu_memory_utilization: float = Field(default=0.90)
    max_model_len: int | None = Field(default=None)
    max_tokens: int = Field(default=512)
    temperature: float = Field(default=0.1)
    top_p: float = Field(default=0.95)
    trust_remote_code: bool = Field(default=False)

    _llm: Any = PrivateAttr(default=None)
    _sampling_params: Any = PrivateAttr(default=None)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.max_model_len or 32768,
            num_output=self.max_tokens,
            model_name=self.model_name,
            is_chat_model=False,
        )

    def _ensure_loaded(self) -> None:
        if self._llm is not None:
            return

        from vllm import LLM, SamplingParams

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.max_model_len:
            kwargs["max_model_len"] = self.max_model_len

        self._llm = LLM(**kwargs)
        self._sampling_params = SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )

    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        self._ensure_loaded()
        sampling_params = self._sampling_params
        if kwargs:
            from vllm import SamplingParams

            sampling_params = SamplingParams(
                max_tokens=int(kwargs.get("max_tokens", self.max_tokens)),
                temperature=float(kwargs.get("temperature", self.temperature)),
                top_p=float(kwargs.get("top_p", self.top_p)),
            )

        outputs = self._llm.generate([prompt], sampling_params)
        text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
        return CompletionResponse(text=text)

    def stream_complete(
        self,
        prompt: str,
        formatted: bool = False,
        **kwargs: Any,
    ) -> Generator[CompletionResponse, None, None]:
        yield self.complete(prompt, formatted=formatted, **kwargs)


class VLLMOfflineEmbedding(BaseEmbedding):
    model_name: str = Field(default="Qwen/Qwen3-Embedding-0.6B")
    tensor_parallel_size: int = Field(default=1)
    gpu_memory_utilization: float = Field(default=0.20)
    max_model_len: int | None = Field(default=None)
    trust_remote_code: bool = Field(default=False)

    _llm: Any = PrivateAttr(default=None)
    _tokenizer: Any = PrivateAttr(default=None)
    _transformer: Any = PrivateAttr(default=None)
    _device: str = PrivateAttr(default="cpu")
    _use_transformers: bool = PrivateAttr(default=False)

    def _ensure_loaded(self) -> None:
        if self._llm is not None or self._transformer is not None:
            return

        try:
            vllm_version = version("vllm")
        except PackageNotFoundError:
            vllm_version = ""

        # vLLM 0.6 exposes encode(), but its registry cannot load BertModel-based
        # encoders such as BGE and E5. Use the already-installed Transformers stack
        # for embeddings while retaining vLLM for generation.
        self._use_transformers = vllm_version.startswith("0.6.")
        if self._use_transformers:
            self._ensure_transformer_loaded()
            return

        from vllm import LLM

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "task": "embed",
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.max_model_len:
            kwargs["max_model_len"] = self.max_model_len
        self._llm = LLM(**kwargs)

    def _ensure_transformer_loaded(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        requested_device = os.environ.get("VLLM_EMBEDDING_FALLBACK_DEVICE", "cuda").strip().lower()
        self._device = "cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )
        self._transformer = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )
        self._transformer.to(self._device)
        self._transformer.eval()

    def _transformer_embed(self, texts: list[str]) -> list[Embedding]:
        import torch
        import torch.nn.functional as functional

        max_length = self.max_model_len or getattr(self._tokenizer, "model_max_length", 512)
        if not isinstance(max_length, int) or max_length > 100_000:
            max_length = 512
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        with torch.inference_mode():
            hidden = self._transformer(**encoded).last_hidden_state
            if "bge-" in self.model_name.lower():
                pooled = hidden[:, 0]
            else:
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = functional.normalize(pooled, p=2, dim=1)
        return pooled.detach().cpu().float().tolist()

    def _extract_embedding(self, output: Any) -> Embedding:
        value = getattr(output, "outputs", output)
        if hasattr(value, "embedding"):
            return list(value.embedding)
        if isinstance(value, dict) and "embedding" in value:
            return list(value["embedding"])
        if isinstance(value, list) and value and isinstance(value[0], (float, int)):
            return list(value)
        if isinstance(value, list) and value and hasattr(value[0], "embedding"):
            return list(value[0].embedding)
        raise TypeError(f"Could not extract embedding from vLLM output type {type(output)!r}.")

    def _embed(self, texts: list[str]) -> list[Embedding]:
        self._ensure_loaded()
        if self._use_transformers:
            return self._transformer_embed(texts)
        if hasattr(self._llm, "embed"):
            outputs = self._llm.embed(texts)
        elif hasattr(self._llm, "encode"):
            outputs = self._llm.encode(texts)
        else:
            raise RuntimeError(
                "This vLLM installation exposes neither LLM.embed() nor LLM.encode() "
                "for offline text embeddings."
            )
        return [self._extract_embedding(output) for output in outputs]

    def _get_text_embedding(self, text: str) -> Embedding:
        return self._embed([text])[0]

    def _get_query_embedding(self, query: str) -> Embedding:
        return self._embed([query])[0]

    async def _aget_query_embedding(self, query: str) -> Embedding:
        return self._get_query_embedding(query)

    def _get_text_embeddings(self, texts: list[str]) -> list[Embedding]:
        return self._embed(texts)


def build_offline_llm(model_name: str | None = None) -> VLLMOfflineLLM:
    return VLLMOfflineLLM(
        model_name=model_name or os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        tensor_parallel_size=int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "1")),
        gpu_memory_utilization=float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.90")),
        max_model_len=(
            int(os.environ["VLLM_MAX_MODEL_LEN"])
            if os.environ.get("VLLM_MAX_MODEL_LEN")
            else None
        ),
        max_tokens=int(os.environ.get("VLLM_MAX_TOKENS", "512")),
        temperature=float(os.environ.get("VLLM_TEMPERATURE", "0.1")),
        top_p=float(os.environ.get("VLLM_TOP_P", "0.95")),
        trust_remote_code=os.environ.get("VLLM_TRUST_REMOTE_CODE", "false").strip().lower()
        in {"1", "true", "yes", "on"},
    )


def build_offline_embedding(model_name: str | None = None) -> VLLMOfflineEmbedding:
    return VLLMOfflineEmbedding(
        model_name=model_name or os.environ.get("VLLM_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B"),
        tensor_parallel_size=int(os.environ.get("VLLM_EMBEDDING_TENSOR_PARALLEL_SIZE", "1")),
        gpu_memory_utilization=float(os.environ.get("VLLM_EMBEDDING_GPU_MEMORY_UTILIZATION", "0.20")),
        max_model_len=(
            int(os.environ["VLLM_EMBEDDING_MAX_MODEL_LEN"])
            if os.environ.get("VLLM_EMBEDDING_MAX_MODEL_LEN")
            else None
        ),
        trust_remote_code=os.environ.get("VLLM_TRUST_REMOTE_CODE", "false").strip().lower()
        in {"1", "true", "yes", "on"},
    )
