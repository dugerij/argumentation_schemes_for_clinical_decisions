import asyncio
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from llama_index.core.graph_stores.simple_labelled import SimplePropertyGraphStore
from llama_index.core.graph_stores.types import ChunkNode, EntityNode, Relation

import retrieval.index as index_module
from retrieval.property_graph import ClinicalEntityNode, ClinicalPropertyGraphStore
from retrieval.index import (
    _PersistentAsyncRunner,
    SchemaGuidanceDocResult,
    _create_empty_index,
    _graph_counts,
    _run_coro_blocking,
    completed_doc_ids,
    init_schema_guidance_db,
    init_checkpoint_db,
    load_cached_schema_guidance,
    pending_documents,
    mark_status,
    source_documents,
    source_file_fingerprint,
    upsert_schema_guidance_result,
)


async def _sample_coro(value: str) -> str:
    await asyncio.sleep(0)
    return value


async def _loop_identity() -> int:
    await asyncio.sleep(0)
    return id(asyncio.get_running_loop())


def test_run_coro_blocking_without_running_loop():
    assert _run_coro_blocking(_sample_coro("ok")) == "ok"


def test_run_coro_blocking_with_running_loop():
    async def inner():
        return _run_coro_blocking(_sample_coro("nested"))

    assert asyncio.run(inner()) == "nested"


def test_persistent_async_runner_reuses_background_loop():
    runner = _PersistentAsyncRunner()
    try:
        first_loop = runner.run(_loop_identity())
        second_loop = runner.run(_loop_identity())
    finally:
        runner.close()

    assert first_loop == second_loop


def test_persistent_async_runner_works_from_running_loop():
    runner = _PersistentAsyncRunner()

    async def inner():
        return runner.run(_sample_coro("nested"))

    try:
        assert asyncio.run(inner()) == "nested"
    finally:
        runner.close()


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


def test_clinical_property_graph_store_merges_entity_and_relation_evidence():
    store = ClinicalPropertyGraphStore()
    creatinine = ClinicalEntityNode(
        name="Creatinine",
        label="LAB_TEST",
        canonical_id="lab_test:creatinine",
        properties={"mention_count": 1, "source_names": ["note-a.txt"]},
    )
    aki = ClinicalEntityNode(
        name="Acute kidney injury",
        label="DISEASE",
        canonical_id="disease:acute_kidney_injury",
        properties={"mention_count": 1, "source_names": ["note-a.txt"]},
    )

    store.upsert_nodes([creatinine, aki])
    store.upsert_nodes(
        [
            ClinicalEntityNode(
                name="Creatinine",
                label="LAB_TEST",
                canonical_id="lab_test:creatinine",
                properties={"mention_count": 1, "source_names": ["note-b.txt"]},
            )
        ]
    )
    store.upsert_relations(
        [
            Relation(
                label="MENTIONS",
                source_id=creatinine.id,
                target_id=aki.id,
                properties={
                    "evidence_count": 1,
                    "source_names": ["note-a.txt"],
                    "evidence_samples": ["Creatinine elevated with AKI."],
                },
            )
        ]
    )
    store.upsert_relations(
        [
            Relation(
                label="MENTIONS",
                source_id=creatinine.id,
                target_id=aki.id,
                properties={
                    "evidence_count": 1,
                    "source_names": ["note-b.txt"],
                    "evidence_samples": ["Creatinine remained elevated."],
                },
            )
        ]
    )

    [merged_creatinine] = store.get(ids=[creatinine.id])
    triplets = store.get_triplets(ids=[creatinine.id])

    assert merged_creatinine.properties["mention_count"] == 2
    assert set(merged_creatinine.properties["source_names"]) == {"note-a.txt", "note-b.txt"}
    assert len(triplets) == 1
    assert triplets[0][1].properties["evidence_count"] == 2
    assert set(triplets[0][1].properties["source_names"]) == {"note-a.txt", "note-b.txt"}


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


def test_pending_documents_skips_completed_checkpoint_entries(tmp_path):
    input_dir = tmp_path / "evidence"
    input_dir.mkdir()
    note1 = input_dir / "note1.txt"
    note2 = input_dir / "note2.txt"
    note1.write_text("one", encoding="utf-8")
    note2.write_text("two", encoding="utf-8")

    conn = init_checkpoint_db(tmp_path / "index_checkpoints.sqlite")
    doc_ids_by_path = {
        note1.as_posix(): source_file_fingerprint(note1, input_dir),
        note2.as_posix(): source_file_fingerprint(note2, input_dir),
    }
    mark_status(conn, doc_ids_by_path[note1.as_posix()], "done")
    mark_status(conn, doc_ids_by_path[note2.as_posix()], "in_progress")

    pending = pending_documents([note1, note2], doc_ids_by_path, conn)

    assert completed_doc_ids(conn) == {doc_ids_by_path[note1.as_posix()]}
    assert pending == [note2]


def test_create_empty_index_assigns_llm_for_implicit_mode():
    fake_llm = object()
    fake_embed_model = object()
    created = {}

    class FakePropertyGraphIndex:
        def __init__(self, **kwargs):
            created.update(kwargs)

    with patch.object(index_module, "build_llm", return_value=fake_llm), patch.object(
        index_module, "build_embed_model", return_value=fake_embed_model
    ), patch.object(index_module, "env_bool", return_value=False), patch(
        "llama_index.core.indices.property_graph.PropertyGraphIndex", FakePropertyGraphIndex
    ):
        _create_empty_index(schema_guided=False)

    assert created["llm"] is fake_llm
    assert created["embed_model"] is fake_embed_model
    assert isinstance(created["storage_context"].property_graph_store, ClinicalPropertyGraphStore)
