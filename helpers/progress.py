import os
import sys
from collections.abc import Iterable, Iterator
from typing import TypeVar

from tqdm.auto import tqdm

from helpers.config import env_bool

T = TypeVar("T")

PROGRESS_ENV_VAR = "SHOW_PROGRESS"


def progress_enabled(default: bool = True) -> bool:
    if os.environ.get(PROGRESS_ENV_VAR) is None:
        return default and sys.stderr.isatty()
    return env_bool(PROGRESS_ENV_VAR, default=default)


def iter_progress(
    iterable: Iterable[T],
    *,
    desc: str,
    total: int | None = None,
    unit: str = "item",
    leave: bool = False,
    enabled: bool | None = None,
) -> Iterator[T]:
    yield from tqdm(
        iterable,
        desc=desc,
        total=total,
        unit=unit,
        leave=leave,
        disable=not progress_enabled() if enabled is None else not enabled,
    )


def progress_message(message: str, *, enabled: bool | None = None) -> None:
    if not (progress_enabled() if enabled is None else enabled):
        print(message)
        return
    tqdm.write(message)
