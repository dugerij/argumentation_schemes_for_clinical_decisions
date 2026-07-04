import asyncio
from types import SimpleNamespace
from pathlib import Path

from llama_index.core.graph_stores.simple_labelled import SimplePropertyGraphStore
from llama_index.core.graph_stores.types import ChunkNode, EntityNode, Relation

from retrieval.index import (
    SchemaGuidanceDocResult,
    _graph_counts,
    _run_coro_blocking,
    init_schema_guidance_db,
    load_cached_schema_guidance,
    source_documents,
    source_file_fingerprint,
    upsert_schema_guidance_result,
)


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


def test_graph_counts_reports_nodes_relations_and_triplets():
    store = SimplePropertyGraphStore()
    chunk = ChunkNode(text="blood pressure stable", id_="chunk-1")
    disease = EntityNode(name="hypertension")
    medication = EntityNode(name="amlodipine")

    store.upsert_nodes([chunk, disease, medication])
    store.upsert_relations(
        [
            Relation(label="MENTIONS", source_id=chunk.id, target_id=disease.id),
            Relation(label="TREATS", source_id=medication.id, target_id=disease.id),
        ]
    )

    stats = _graph_counts(SimpleNamespace(property_graph_store=store))

    assert stats is not None
    assert stats.total_nodes == 3
    assert stats.entity_nodes == 2
    assert stats.chunk_nodes == 1
    assert stats.relation_count == 2
    assert stats.triplet_count == 2


def test_schema_guidance_cache_reuses_matching_fingerprints(tmp_path):
    input_dir = tmp_path / "evidence"
    input_dir.mkdir()
    note = input_dir / "note1.txt"
    note.write_text("hypertension amlodipine", encoding="utf-8")

    conn = init_schema_guidance_db(tmp_path / "schema_guidance.sqlite")
    result = SchemaGuidanceDocResult(
        path=note.as_posix(),
        source_fingerprint=source_file_fingerprint(note, input_dir),
        source_name=note.name,
        char_count=23,
        candidate_term_count=2,
        mention_count=2,
        entity_types=("DISEASE", "MEDICATION"),
        hint_block="UMLS concept hints:",
        duration_seconds=0.12,
    )
    upsert_schema_guidance_result(conn, result)

    cached = load_cached_schema_guidance(conn, [note], input_dir)

    assert note.as_posix() in cached
    assert cached[note.as_posix()].mention_count == 2
    assert cached[note.as_posix()].entity_types == ("DISEASE", "MEDICATION")


def test_schema_guidance_cache_ignores_stale_fingerprints(tmp_path):
    input_dir = tmp_path / "evidence"
    input_dir.mkdir()
    note = input_dir / "note1.txt"
    note.write_text("hypertension", encoding="utf-8")

    conn = init_schema_guidance_db(tmp_path / "schema_guidance.sqlite")
    result = SchemaGuidanceDocResult(
        path=note.as_posix(),
        source_fingerprint="stale",
        source_name=note.name,
        char_count=12,
        candidate_term_count=1,
        mention_count=1,
        entity_types=("DISEASE",),
        hint_block="UMLS concept hints:",
        duration_seconds=0.1,
    )
    upsert_schema_guidance_result(conn, result)

    cached = load_cached_schema_guidance(conn, [note], input_dir)

    assert cached == {}
