import asyncio
from pathlib import Path

from retrieval.index import _run_coro_blocking, source_documents


async def _sample_coro(value: str) -> str:
    await asyncio.sleep(0)
    return value


def test_run_coro_blocking_without_running_loop():
    assert _run_coro_blocking(_sample_coro("ok")) == "ok"


def test_run_coro_blocking_with_running_loop():
    async def inner():
        return _run_coro_blocking(_sample_coro("nested"))

    assert asyncio.run(inner()) == "nested"


def test_source_documents_only_counts_extracted_txt_files(tmp_path):
    input_dir = tmp_path / "evidence"
    input_dir.mkdir()
    (input_dir / "note1.txt").write_text("note 1", encoding="utf-8")
    (input_dir / "note2.txt").write_text("note 2", encoding="utf-8")
    (input_dir / "manifest.json").write_text("{}", encoding="utf-8")
    nested = input_dir / "nested"
    nested.mkdir()
    (nested / "note3.txt").write_text("note 3", encoding="utf-8")
    (nested / "ignore.csv").write_text("id,text\n1,x\n", encoding="utf-8")

    documents = source_documents(input_dir)

    assert [path.relative_to(input_dir).as_posix() for path in documents] == [
        "nested/note3.txt",
        "note1.txt",
        "note2.txt",
    ]
