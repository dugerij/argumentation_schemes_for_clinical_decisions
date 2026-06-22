import os
import json


VALID_PROVIDERS = {'ollama', 'openai', 'gemini', 'together_ai', 'vllm'}
TRUTHY = {'1', 'true', 'yes', 'on'}


def _require_any(primary: str, fallback: str | None = None) -> None:
    if os.environ.get(primary):
        return
    if fallback and os.environ.get(fallback):
        return
    names = f"{primary} or {fallback}" if fallback else primary
    raise ValueError(f"Missing required environment variable: {names}")


def get_model_name(role: str) -> str:
    role = role.upper()
    return os.environ.get(f"{role}_MODEL") or os.environ[f"{role}_BASE_MODEL"]


def umls_enabled() -> bool:
    return os.environ.get("UMLS_ENABLED", "false").strip().lower() in TRUTHY


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in TRUTHY


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return int(value)


def parse_optional_int(value: str | None, default: int | None = None) -> int | None:
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"all", "none", "null", "unlimited"}:
        return None
    return int(value)


def env_optional_int(name: str, default: int | None = None) -> int | None:
    return parse_optional_int(os.environ.get(name), default=default)


def startup_check():
    """"Checks that all required environment variables are set and valid
    Raises:
        ValueError: If any required environment variable is missing or invalid.
    """
    required_env_vars = [
        'GENERATION_MODEL_PROVIDER',
        'VERIFIER_MODEL_PROVIDER',
        'REASONER_MODEL_PROVIDER',
        'RAG_EMBEDDING_MODEL_PROVIDER',
        'RAG_EMBEDDING_MODEL',
        'INPUT_BASE_DIR',
        'OUTPUT_BASE_DIR',
    ]
    
    for var in required_env_vars:
        if var not in os.environ:
            raise ValueError(f"Missing required environment variable: {var}")

    _require_any('GENERATOR_MODEL', 'GENERATOR_BASE_MODEL')
    _require_any('VERIFIER_MODEL', 'VERIFIER_BASE_MODEL')
    _require_any('REASONER_MODEL', 'REASONER_BASE_MODEL')
    if umls_enabled() and not os.environ.get('UMLS_API_KEY'):
        raise ValueError("UMLS_API_KEY must be set when UMLS_ENABLED=true.")
            
    # Validate providers are valid values
    providers = [
        os.environ.get('GENERATION_MODEL_PROVIDER'),
        os.environ.get('VERIFIER_MODEL_PROVIDER'),
        os.environ.get('REASONER_MODEL_PROVIDER'),
        os.environ.get('RAG_EMBEDDING_MODEL_PROVIDER'),
    ]
    for provider in providers:
        if provider not in VALID_PROVIDERS:
            raise ValueError("Invalid provider specified. Please choose from 'ollama', 'together_ai', 'openai', 'gemini', or 'vllm'.")

    # Validate provider-specific credentials
    provider_vars = [
        'GENERATION_MODEL_PROVIDER',
        'VERIFIER_MODEL_PROVIDER',
        'REASONER_MODEL_PROVIDER',
        'RAG_EMBEDDING_MODEL_PROVIDER',
    ]
    for var in provider_vars:
        provider = os.environ.get(var)
        if provider == 'openai' and not os.environ.get('OPENAI_API_KEY'):
            raise ValueError("OPENAI_API_KEY must be set when using OpenAI as a provider.")
        elif provider == 'gemini' and not os.environ.get('GEMINI_API_KEY'):
            raise ValueError("GEMINI_API_KEY must be set when using Gemini as a provider.")
        elif provider == 'together_ai' and not os.environ.get('GRAPHRAG_API_KEY'):
            raise ValueError("GRAPHRAG_API_KEY must be set when using Together AI as a provider.")
            
    print("Startup check passed. All required environment variables are set and valid.")


def load_dataset(file_path: str):
    with open(file_path, 'r') as file:
        for line in file:
            raw_json = json.loads(line)
            print(raw_json)
