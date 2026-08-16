import json
import time
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id(prefix: str = "run") -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{timestamp}_{uuid.uuid4().hex[:8]}"


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    return value


class JsonlLogger:
    def __init__(self, path: Path, run_id: str | None = None):
        self.path = path
        self.run_id = run_id or new_run_id()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, step: str, status: str, **payload: Any) -> None:
        record = {
            "timestamp": utc_now(),
            "run_id": self.run_id,
            "step": step,
            "status": status,
            **to_jsonable(payload),
        }
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def timed(self, step: str, **payload: Any):
        return TimedEvent(self, step, payload)


class TimedEvent:
    def __init__(self, logger: JsonlLogger, step: str, payload: dict[str, Any]):
        self.logger = logger
        self.step = step
        self.payload = payload
        self.started_at = 0.0

    def __enter__(self):
        self.started_at = time.perf_counter()
        self.logger.event(self.step, "started", **self.payload)
        return self

    def __exit__(self, exc_type, exc, _traceback):
        duration_ms = round((time.perf_counter() - self.started_at) * 1000, 2)
        if exc is None:
            self.logger.event(self.step, "completed", duration_ms=duration_ms, **self.payload)
            return False

        self.logger.event(
            self.step,
            "failed",
            duration_ms=duration_ms,
            error_type=exc_type.__name__ if exc_type else None,
            error=str(exc),
            **self.payload,
        )
        return False


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(to_jsonable(record), ensure_ascii=False) + "\n")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(to_jsonable(record), ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]
