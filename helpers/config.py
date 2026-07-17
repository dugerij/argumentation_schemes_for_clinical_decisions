import json
import os

from helpers.experiment_models import DEFAULT_EMBEDDING_MODEL, DEFAULT_GENERATION_MODEL

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


def startup_check(*, require_models: bool = True, require_paths: bool = True):
    if require_paths:
        for var in ("OUTPUT_BASE_DIR",):
            if var not in os.environ:
                raise ValueError(f"Missing required environment variable: {var}")

    if require_models:
        os.environ.setdefault('GENERATION_MODEL_PROVIDER', 'ollama')
        os.environ.setdefault('VERIFIER_MODEL_PROVIDER', 'ollama')
        os.environ.setdefault('REASONER_MODEL_PROVIDER', 'ollama')
        os.environ.setdefault('RAG_EMBEDDING_MODEL_PROVIDER', 'ollama')
        os.environ.setdefault('GENERATOR_MODEL', DEFAULT_GENERATION_MODEL)
        os.environ.setdefault('VERIFIER_MODEL', os.environ['GENERATOR_MODEL'])
        os.environ.setdefault('REASONER_MODEL', os.environ['GENERATOR_MODEL'])
        os.environ.setdefault('RAG_EMBEDDING_MODEL', DEFAULT_EMBEDDING_MODEL)

        _require_any('GENERATOR_MODEL', 'GENERATOR_BASE_MODEL')
        _require_any('VERIFIER_MODEL', 'VERIFIER_BASE_MODEL')
        _require_any('REASONER_MODEL', 'REASONER_BASE_MODEL')

        provider_vars = (
            'GENERATION_MODEL_PROVIDER',
            'VERIFIER_MODEL_PROVIDER',
            'REASONER_MODEL_PROVIDER',
            'RAG_EMBEDDING_MODEL_PROVIDER',
        )
        for var in provider_vars:
            if os.environ.get(var, 'ollama').strip().lower() != 'ollama':
                raise ValueError(f"{var} must be 'ollama'. This repo now supports only Ollama-backed models.")

    print("Startup check passed. All required environment variables are set and valid.")


def load_dataset(file_path: str):
    with open(file_path, 'r') as file:
        for line in file:
            raw_json = json.loads(line)
            print(raw_json)
