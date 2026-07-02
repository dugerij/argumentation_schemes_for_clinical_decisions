GENERATION_MODEL_SWEEP = (
    "gemma4:latest",
    "qwen3.5:9b",
    "medgemma1.5:latest",
)

EMBEDDING_MODEL_SWEEP = (
    "qwen3-embedding:0.6b",
    "embeddinggemma:latest",
    "all-minilm:latest",
)

DEFAULT_GENERATION_MODEL = GENERATION_MODEL_SWEEP[0]
DEFAULT_EMBEDDING_MODEL = EMBEDDING_MODEL_SWEEP[0]
