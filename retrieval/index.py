import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import sqlite3
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from helpers.config import env_bool, env_int
from helpers.jsonl import JsonlLogger, new_run_id
from helpers.ollama import assert_ollama_available, ollama_endpoint
from helpers.paths import INDEX_BUILD_LOG_PATH
from helpers.progress import iter_progress, progress_enabled, progress_message
from retrieval.concepts.candidates import extract_candidate_terms
from retrieval.concepts.extractor import UMLSConceptExtractor
from retrieval.concepts.medical_schema import (
    ClinicalSchemaGuidance,
    MedicalEntityType,
    MedicalRelationType,
    ENTITY_LABELS,
    RELATION_LABELS,
    build_validation_schema,
    format_concept_hint_block,
    entity_type_for_category,
)
from retrieval.concepts.umls import UMLSConfig, create_umls_client


INDEX_MANIFEST = "index_manifest.json"
CHECKPOINT_DB = "index_checkpoints.sqlite"
SCHEMA_GUIDANCE_DB = "schema_guidance.sqlite"
SOURCE_FILE_PATTERN = "*.txt"
DEFAULT_BATCH_SIZE = 20
DEFAULT_MAX_RETRIES = 3
DEFAULT_SCHEMA_HIT_LIMIT = 120
INDEX_EVENT_LOG = INDEX_BUILD_LOG_PATH
DEFAULT_LLM_REQUEST_TIMEOUT = 180.0


@dataclass(frozen=True)
class GraphCounts:
    total_nodes: int
    entity_nodes: int
    chunk_nodes: int
    relation_count: int
    triplet_count: int


@dataclass(frozen=True)
class SchemaGuidanceDocResult:
    path: str
    source_fingerprint: str
    source_name: str
    char_count: int
    candidate_term_count: int
    mention_count: int
    entity_types: tuple[str, ...]
    hint_block: str
    duration_seconds: float


_SCHEMA_GUIDANCE_WORKER: dict[str, Any] = {}


def _run_coro_blocking(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, Any] = {}
    error: dict[str, BaseException] = {}

    def runner() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # pragma: no cover - surfaced to caller
            error["exc"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if "exc" in error:
        raise error["exc"]
    return result.get("value")


def source_documents(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        return []
    return sorted(path for path in input_dir.rglob(SOURCE_FILE_PATTERN) if path.is_file())


def fingerprint_documents(documents: list[Path], input_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in documents:
        relative = path.relative_to(input_dir)
        stat = path.stat()
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        digest.update(f"{relative.as_posix()}|{stat.st_size}|{stat.st_mtime_ns}|{file_hash}\n".encode())
    return digest.hexdigest()


def assert_source_documents(input_dir: Path) -> None:
    documents = source_documents(input_dir)
    if documents:
        return

    raise FileNotFoundError(
        f"No {SOURCE_FILE_PATTERN} source documents were found in {input_dir}.\n"
        "Populate the evidence folder with extracted clinical notes first. "
        "You can run `python make_index.py extract-mimic-discharge --limit all --max-chars all`."
    )


def ensure_source_documents(input_dir: Path) -> list[Path]:
    documents = source_documents(input_dir)
    if documents:
        return documents

    assert_source_documents(input_dir)
    return documents


def init_checkpoint_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS indexed_docs (
            doc_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            error TEXT,
            updated_at REAL NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def init_schema_guidance_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_guidance (
            source_path TEXT PRIMARY KEY,
            source_fingerprint TEXT NOT NULL,
            source_name TEXT NOT NULL,
            char_count INTEGER NOT NULL,
            candidate_term_count INTEGER NOT NULL,
            mention_count INTEGER NOT NULL,
            entity_types_json TEXT NOT NULL,
            hint_block TEXT NOT NULL,
            duration_seconds REAL NOT NULL,
            updated_at REAL NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def load_cached_schema_guidance(
    conn: sqlite3.Connection,
    documents: list[Path],
    input_dir: Path,
) -> dict[str, SchemaGuidanceDocResult]:
    cached: dict[str, SchemaGuidanceDocResult] = {}
    fingerprints = {path.as_posix(): source_file_fingerprint(path, input_dir) for path in documents}
    if not fingerprints:
        return cached

    source_paths = list(fingerprints.keys())
    chunk_size = 900
    for start in range(0, len(source_paths), chunk_size):
        chunk = source_paths[start : start + chunk_size]
        placeholders = ",".join("?" for _ in chunk)
        rows = conn.execute(
            f"""
            SELECT
                source_path,
                source_fingerprint,
                source_name,
                char_count,
                candidate_term_count,
                mention_count,
                entity_types_json,
                hint_block,
                duration_seconds
            FROM schema_guidance
            WHERE source_path IN ({placeholders})
            """,
            chunk,
        ).fetchall()
        for row in rows:
            if fingerprints.get(row[0]) != row[1]:
                continue
            cached[row[0]] = SchemaGuidanceDocResult(
                path=row[0],
                source_fingerprint=row[1],
                source_name=row[2],
                char_count=row[3],
                candidate_term_count=row[4],
                mention_count=row[5],
                entity_types=tuple(json.loads(row[6])),
                hint_block=row[7],
                duration_seconds=row[8],
            )
    return cached


def upsert_schema_guidance_result(conn: sqlite3.Connection, result: SchemaGuidanceDocResult) -> None:
    conn.execute(
        """
        INSERT INTO schema_guidance(
            source_path,
            source_fingerprint,
            source_name,
            char_count,
            candidate_term_count,
            mention_count,
            entity_types_json,
            hint_block,
            duration_seconds,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_path)
        DO UPDATE SET source_fingerprint = excluded.source_fingerprint,
                      source_name = excluded.source_name,
                      char_count = excluded.char_count,
                      candidate_term_count = excluded.candidate_term_count,
                      mention_count = excluded.mention_count,
                      entity_types_json = excluded.entity_types_json,
                      hint_block = excluded.hint_block,
                      duration_seconds = excluded.duration_seconds,
                      updated_at = excluded.updated_at
        """,
        (
            result.path,
            result.source_fingerprint,
            result.source_name,
            result.char_count,
            result.candidate_term_count,
            result.mention_count,
            json.dumps(list(result.entity_types)),
            result.hint_block,
            result.duration_seconds,
            time.time(),
        ),
    )
    conn.commit()


def already_done(conn: sqlite3.Connection, doc_id: str) -> bool:
    row = conn.execute(
        "SELECT status FROM indexed_docs WHERE doc_id = ?",
        (doc_id,),
    ).fetchone()
    return row is not None and row[0] == "done"


def mark_status(conn: sqlite3.Connection, doc_id: str, status: str, error: str | None = None) -> None:
    conn.execute(
        """
        INSERT INTO indexed_docs(doc_id, status, error, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(doc_id)
        DO UPDATE SET status = excluded.status,
                      error = excluded.error,
                      updated_at = excluded.updated_at
        """,
        (doc_id, status, error, time.time()),
    )
    conn.commit()


def file_doc_id(path: Path, input_dir: Path) -> str:
    relative = path.relative_to(input_dir).as_posix()
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashlib.sha256(f"{relative}|{file_hash}".encode()).hexdigest()


def source_file_fingerprint(path: Path, input_dir: Path) -> str:
    relative = path.relative_to(input_dir).as_posix()
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashlib.sha256(f"{relative}|{file_hash}".encode()).hexdigest()


def read_source_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def build_document(path: Path, input_dir: Path, hint_block: str | None = None):
    from llama_index.core import Document

    doc_id = file_doc_id(path, input_dir)
    text = read_source_file(path)
    if hint_block:
        text = f"{hint_block}\n\nSOURCE TEXT:\n{text}"
    metadata = {
        "source_path": path.as_posix(),
        "source_name": path.name,
        "source_type": "clinical_note",
    }

    try:
        return Document(text=text, id_=doc_id, metadata=metadata)
    except TypeError:
        return Document(text=text, doc_id=doc_id, metadata=metadata)


def _schema_guided_enabled() -> bool:
    return env_bool("INDEX_SCHEMA_GUIDED", True)


def _umls_client():
    if not env_bool("UMLS_ENABLED", False):
        return None
    return create_umls_client(UMLSConfig.from_env())


def _dedupe_mentions(mentions):
    deduped = []
    seen: set[tuple[str | None, int | None, int | None]] = set()
    for mention in mentions:
        concept = mention.concept
        key = (
            concept.cui if concept else mention.text.lower(),
            mention.start_char,
            mention.end_char,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(mention)
    return deduped


def _graph_counts(index: Any) -> GraphCounts | None:
    store = getattr(index, "property_graph_store", None)
    graph = getattr(store, "graph", None)
    if graph is None:
        return None

    nodes = list(getattr(graph, "nodes", {}).values())
    relations = getattr(graph, "relations", {})
    triplets = getattr(graph, "triplets", set())
    return GraphCounts(
        total_nodes=len(nodes),
        entity_nodes=sum(1 for node in nodes if getattr(node, "label", None) == "entity"),
        chunk_nodes=sum(1 for node in nodes if getattr(node, "label", None) == "text_chunk"),
        relation_count=len(relations),
        triplet_count=len(triplets),
    )


def _format_graph_counts(counts: GraphCounts) -> str:
    return (
        f"{counts.total_nodes} nodes "
        f"({counts.entity_nodes} entities, {counts.chunk_nodes} chunks), "
        f"{counts.relation_count} relationships, {counts.triplet_count} triplets"
    )


def _format_graph_delta(before: GraphCounts | None, after: GraphCounts | None) -> str:
    if before is None or after is None:
        return ""

    return (
        f" | graph +{after.total_nodes - before.total_nodes} nodes, "
        f"+{after.relation_count - before.relation_count} relationships "
        f"(now {_format_graph_counts(after)})"
    )


def _init_schema_guidance_worker() -> None:
    client = _umls_client()
    if client is None:
        raise RuntimeError("UMLS guidance worker could not initialize a client.")
    _SCHEMA_GUIDANCE_WORKER["extractor"] = UMLSConceptExtractor(client)


def _build_schema_guidance_for_document(path: Path, input_dir: Path, limit: int) -> SchemaGuidanceDocResult:
    extractor: UMLSConceptExtractor | None = _SCHEMA_GUIDANCE_WORKER.get("extractor")
    if extractor is None:
        client = _umls_client()
        if client is None:
            raise RuntimeError("UMLS guidance is enabled but no UMLS client is configured.")
        extractor = UMLSConceptExtractor(client)
        _SCHEMA_GUIDANCE_WORKER["extractor"] = extractor

    started_at = time.perf_counter()
    text = read_source_file(path)
    candidate_terms = extract_candidate_terms(text, limit=limit)
    mentions = _dedupe_mentions(extractor.extract_from_terms(text, candidate_terms, max_mentions=20))
    entity_types = tuple(
        sorted(
            {
                entity_type_for_category(mention.category or (mention.concept.category if mention.concept else None))
                for mention in mentions
                if entity_type_for_category(mention.category or (mention.concept.category if mention.concept else None))
            }
        )
    )
    hint_block = format_concept_hint_block(mentions, max_mentions=20, max_relations=20)
    return SchemaGuidanceDocResult(
        path=path.as_posix(),
        source_fingerprint=source_file_fingerprint(path, input_dir),
        source_name=path.name,
        char_count=len(text),
        candidate_term_count=len(candidate_terms),
        mention_count=len(mentions),
        entity_types=entity_types,
        hint_block=hint_block,
        duration_seconds=round(time.perf_counter() - started_at, 2),
    )


def build_schema_guidance(
    documents: list[Path],
    input_dir: Path,
    output_dir: Path,
    *,
    event_logger: JsonlLogger | None = None,
) -> ClinicalSchemaGuidance:
    client = _umls_client()
    if client is None:
        validation_schema = build_validation_schema(set(ENTITY_LABELS))
        return ClinicalSchemaGuidance(
            validation_schema=validation_schema,
            concept_hints_by_source={},
            concept_count_by_source={},
            candidate_count_by_source={},
            relation_count=len(validation_schema),
        )

    hint_by_source: dict[str, str] = {}
    counts_by_source: dict[str, int] = {}
    candidate_counts_by_source: dict[str, int] = {}
    observed_entity_types: set[str] = set()

    limit = env_int("UMLS_HINT_LIMIT", DEFAULT_SCHEMA_HIT_LIMIT)
    total_documents = len(documents)
    worker_count = max(1, int(os.environ.get("UMLS_GUIDANCE_NUM_WORKERS", "1")))
    cache_conn = init_schema_guidance_db(output_dir / SCHEMA_GUIDANCE_DB)
    cached_results = load_cached_schema_guidance(cache_conn, documents, input_dir)
    pending_documents = [path for path in documents if path.as_posix() not in cached_results]

    progress_message(
        f"UMLS guidance cache: reused {len(cached_results):,} document(s), "
        f"computing {len(pending_documents):,} document(s)"
    )
    if event_logger is not None:
        event_logger.event(
            "schema_guidance_cache",
            "completed",
            cached_document_count=len(cached_results),
            pending_document_count=len(pending_documents),
            cache_db=(output_dir / SCHEMA_GUIDANCE_DB).as_posix(),
        )

    def handle_result(doc_number: int, result: SchemaGuidanceDocResult, *, persist: bool) -> None:
        candidate_counts_by_source[result.path] = result.candidate_term_count
        counts_by_source[result.path] = result.mention_count
        observed_entity_types.update(result.entity_types)
        if result.hint_block:
            hint_by_source[result.path] = result.hint_block
        if persist:
            upsert_schema_guidance_result(cache_conn, result)
        progress_message(
            f"UMLS guidance {doc_number}/{total_documents} completed: {result.source_name} "
            f"-> {result.mention_count} concepts in {result.duration_seconds}s"
        )
        if event_logger is not None:
            event_logger.event(
                "schema_guidance_document",
                "completed",
                source_path=result.path,
                source_name=result.source_name,
                candidate_term_count=result.candidate_term_count,
                umls_concept_count=result.mention_count,
                has_hint_block=bool(result.hint_block),
                duration_seconds=result.duration_seconds,
            )

    for path in documents:
        cached = cached_results.get(path.as_posix())
        if cached is None:
            continue
        candidate_counts_by_source[cached.path] = cached.candidate_term_count
        counts_by_source[cached.path] = cached.mention_count
        observed_entity_types.update(cached.entity_types)
        if cached.hint_block:
            hint_by_source[cached.path] = cached.hint_block

    if not pending_documents:
        validation_schema = build_validation_schema({entity_type for entity_type in observed_entity_types if entity_type})
        cache_conn.close()
        return ClinicalSchemaGuidance(
            validation_schema=validation_schema,
            concept_hints_by_source=hint_by_source,
            concept_count_by_source=counts_by_source,
            candidate_count_by_source=candidate_counts_by_source,
            relation_count=len(validation_schema),
        )

    completed_so_far = len(cached_results)
    if worker_count == 1:
        for doc_number, path in enumerate(
            iter_progress(
                pending_documents,
                desc="UMLS schema guidance",
                total=len(pending_documents),
                unit="doc",
            ),
            start=1,
        ):
            text = read_source_file(path)
            progress_message(
                f"UMLS guidance {completed_so_far + doc_number}/{total_documents}: {path.name} "
                f"({len(text):,} chars)"
            )
            if event_logger is not None:
                event_logger.event(
                    "schema_guidance_document",
                    "started",
                    source_path=path.as_posix(),
                    source_name=path.name,
                    char_count=len(text),
                )
            result = _build_schema_guidance_for_document(path, input_dir, limit)
            handle_result(completed_so_far + doc_number, result, persist=True)
    else:
        progress_message(f"UMLS guidance parallel workers: {worker_count}")
        if event_logger is not None:
            for path in pending_documents:
                event_logger.event(
                    "schema_guidance_document",
                    "queued",
                    source_path=path.as_posix(),
                    source_name=path.name,
                )
        with ThreadPoolExecutor(max_workers=worker_count, initializer=_init_schema_guidance_worker) as executor:
            futures = {
                executor.submit(_build_schema_guidance_for_document, path, input_dir, limit): path
                for path in pending_documents
            }
            for doc_number, future in enumerate(
                iter_progress(
                    as_completed(futures),
                    desc="UMLS schema guidance",
                    total=len(pending_documents),
                    unit="doc",
                ),
                start=1,
            ):
                result = future.result()
                handle_result(completed_so_far + doc_number, result, persist=True)

    validation_schema = build_validation_schema({entity_type for entity_type in observed_entity_types if entity_type})
    cache_conn.close()
    return ClinicalSchemaGuidance(
        validation_schema=validation_schema,
        concept_hints_by_source=hint_by_source,
        concept_count_by_source=counts_by_source,
        candidate_count_by_source=candidate_counts_by_source,
        relation_count=len(validation_schema),
    )


def build_llm():
    model_name = os.environ.get("INDEX_LLM_MODEL", os.environ.get("GENERATOR_MODEL", "")).strip()
    request_timeout = float(os.environ.get("INDEX_LLM_REQUEST_TIMEOUT", str(DEFAULT_LLM_REQUEST_TIMEOUT)))
    from llama_index.llms.ollama import Ollama

    resolved_model = model_name or "llama3.1"
    progress_message(f"Checking Ollama LLM model '{resolved_model}' at {ollama_endpoint()}")
    assert_ollama_available(role="LLM", model_name=resolved_model)
    progress_message(f"Ollama LLM model reachable: {resolved_model}")
    return Ollama(
        model=resolved_model,
        base_url=ollama_endpoint(),
        request_timeout=request_timeout,
    )


def build_embed_model():
    model_name = os.environ.get("INDEX_EMBEDDING_MODEL", os.environ.get("RAG_EMBEDDING_MODEL", "")).strip()
    from llama_index.embeddings.ollama import OllamaEmbedding

    resolved_model = model_name or "embeddinggemma:300m"
    progress_message(f"Checking Ollama embedding model '{resolved_model}' at {ollama_endpoint()}")
    assert_ollama_available(role="embedding", model_name=resolved_model)
    progress_message(f"Ollama embedding model reachable: {resolved_model}")
    return OllamaEmbedding(
        model_name=resolved_model,
        base_url=ollama_endpoint(),
    )


def manifest_path(output_dir: Path) -> Path:
    return output_dir / INDEX_MANIFEST


def metadata_for_sources(
    input_dir: Path,
    documents: list[Path],
    schema_guided: bool,
    use_umls: bool,
    guidance: ClinicalSchemaGuidance | None = None,
) -> dict[str, object]:
    llm_provider = os.environ.get("INDEX_LLM_PROVIDER", os.environ.get("GENERATION_MODEL_PROVIDER", "ollama")).strip().lower()
    llm_model = os.environ.get("INDEX_LLM_MODEL", os.environ.get("GENERATOR_MODEL", "")).strip()
    embedding_provider = os.environ.get(
        "INDEX_EMBEDDING_PROVIDER",
        os.environ.get("RAG_EMBEDDING_MODEL_PROVIDER", "ollama"),
    ).strip().lower()
    embedding_model = os.environ.get("INDEX_EMBEDDING_MODEL", os.environ.get("RAG_EMBEDDING_MODEL", "")).strip()
    return {
        "backend": "llamaindex",
        "index_mode": "schema_guided" if schema_guided else "implicit",
        "source_dir": input_dir.as_posix(),
        "document_count": len(documents),
        "source_fingerprint": fingerprint_documents(documents, input_dir),
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "entity_types": list(ENTITY_LABELS),
        "relation_types": list(RELATION_LABELS),
        "schema_guided": schema_guided,
        "umls_enabled": use_umls,
        "umls_concept_count": sum((guidance.concept_count_by_source.get(path.as_posix(), 0) if guidance else 0) for path in documents),
        "candidate_term_count": sum((guidance.candidate_count_by_source.get(path.as_posix(), 0) if guidance else 0) for path in documents),
        "relation_hint_count": guidance.relation_count if guidance else 0,
        "updated_at": time.time(),
    }


def index_needs_rebuild(
    output_dir: Path,
    input_dir: Path,
    schema_guided: bool,
    use_umls: bool,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    manifest_file = manifest_path(output_dir)
    documents = source_documents(input_dir)

    if not manifest_file.exists():
        reasons.append(f"missing {manifest_file}")
        return True, reasons

    if not documents:
        reasons.append(f"no {SOURCE_FILE_PATTERN} files found in {input_dir}")
        return True, reasons

    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except Exception as exc:
        reasons.append(f"could not read index manifest: {exc}")
        return True, reasons

    if manifest.get("backend") != "llamaindex":
        reasons.append("index manifest was built by a different backend")
    if manifest.get("index_mode") != ("schema_guided" if schema_guided else "implicit"):
        reasons.append("index mode changed")
    current_llm_provider = os.environ.get("INDEX_LLM_PROVIDER", os.environ.get("GENERATION_MODEL_PROVIDER", "ollama")).strip().lower()
    current_llm_model = os.environ.get("INDEX_LLM_MODEL", os.environ.get("GENERATOR_MODEL", "")).strip()
    current_embedding_provider = os.environ.get(
        "INDEX_EMBEDDING_PROVIDER",
        os.environ.get("RAG_EMBEDDING_MODEL_PROVIDER", "ollama"),
    ).strip().lower()
    current_embedding_model = os.environ.get("INDEX_EMBEDDING_MODEL", os.environ.get("RAG_EMBEDDING_MODEL", "")).strip()
    if schema_guided and (
        manifest.get("llm_provider") != current_llm_provider or manifest.get("llm_model") != current_llm_model
    ):
        reasons.append("LLM model changed")
    if manifest.get("embedding_provider") != current_embedding_provider or manifest.get("embedding_model") != current_embedding_model:
        reasons.append("embedding model changed")
    if bool(manifest.get("umls_enabled", False)) != use_umls:
        reasons.append("UMLS setting changed")

    current_fingerprint = fingerprint_documents(documents, input_dir)
    if manifest.get("source_fingerprint") != current_fingerprint:
        reasons.append("source fingerprint changed")

    return bool(reasons), reasons


def _load_existing_index(output_dir: Path):
    from llama_index.core import Settings, StorageContext, load_index_from_storage

    llm = build_llm()
    embed_model = build_embed_model()
    Settings.llm = llm
    Settings.embed_model = embed_model
    storage_context = StorageContext.from_defaults(persist_dir=str(output_dir))
    try:
        index = load_index_from_storage(
            storage_context,
            llm=llm,
            embed_model=embed_model,
        )
    except Exception:
        from llama_index.core.indices.property_graph import PropertyGraphIndex

        if hasattr(PropertyGraphIndex, "from_existing"):
            index = PropertyGraphIndex.from_existing(
                property_graph_store=storage_context.property_graph_store,
                vector_store=getattr(storage_context, "vector_store", None),
                llm=llm,
                embed_model=embed_model,
            )
        else:
            raise

    if hasattr(index, "_use_async"):
        index._use_async = False
    return index


def _create_empty_index(schema_guided: bool, validation_schema: list[tuple[str, str, str]] | None = None):
    from llama_index.core import StorageContext
    from llama_index.core.indices.property_graph import (
        ImplicitPathExtractor,
        PropertyGraphIndex,
        SchemaLLMPathExtractor,
    )
    from llama_index.core.graph_stores.types import KG_NODES_KEY, KG_RELATIONS_KEY
    from llama_index.core.schema import MetadataMode

    class NotebookSafeSchemaLLMPathExtractor(SchemaLLMPathExtractor):
        def __call__(self, nodes, show_progress: bool = False, **kwargs: Any):
            extracted_nodes = []
            for node in nodes:
                text = node.get_content(metadata_mode=MetadataMode.LLM)
                try:
                    kg_schema = self.llm.structured_predict(
                        self.kg_schema_cls,
                        self.extract_prompt,
                        text=text,
                        max_triplets_per_chunk=self.max_triplets_per_chunk,
                    )
                    triplets = self._prune_invalid_triplets(kg_schema)
                except (ValueError, TypeError, AttributeError):
                    triplets = []

                existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
                existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])
                metadata = node.metadata.copy()

                for subj, rel, obj in triplets:
                    subj.properties.update(metadata)
                    obj.properties.update(metadata)
                    rel.properties.update(metadata)
                    existing_relations.append(rel)
                    existing_nodes.append(subj)
                    existing_nodes.append(obj)

                node.metadata[KG_NODES_KEY] = existing_nodes
                node.metadata[KG_RELATIONS_KEY] = existing_relations
                extracted_nodes.append(node)

            return extracted_nodes

    llm = build_llm()
    embed_model = build_embed_model()
    if schema_guided:
        schema_num_workers = int(os.environ.get("INDEX_SCHEMA_NUM_WORKERS", "1"))
        schema_triplets = int(os.environ.get("INDEX_SCHEMA_MAX_TRIPLETS_PER_CHUNK", "6"))
        kg_extractor = NotebookSafeSchemaLLMPathExtractor(
            llm=llm,
            possible_entities=MedicalEntityType,
            possible_relations=MedicalRelationType,
            strict=True,
            kg_validation_schema=validation_schema or build_validation_schema(set(ENTITY_LABELS)),
            max_triplets_per_chunk=schema_triplets,
            num_workers=schema_num_workers,
        )
    else:
        kg_extractor = ImplicitPathExtractor()
    storage_context = StorageContext.from_defaults()
    return PropertyGraphIndex(
        nodes=[],
        kg_extractors=[kg_extractor],
        storage_context=storage_context,
        llm=llm,
        embed_model=embed_model,
        use_async=False,
        show_progress=progress_enabled(),
    )


def load_index(output_dir: Path):
    return _load_existing_index(output_dir)


def ensure_index(
    input_dir: Path,
    output_dir: Path,
    use_umls: bool | None = None,
    schema_guided: bool | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_retries: int = DEFAULT_MAX_RETRIES,
):
    if use_umls is None:
        use_umls = env_bool("UMLS_ENABLED", False)
    if schema_guided is None:
        schema_guided = _schema_guided_enabled()

    run_id = new_run_id("index_build")
    event_logger = JsonlLogger(INDEX_EVENT_LOG, run_id=run_id)
    progress_message(f"Detailed build log: {INDEX_EVENT_LOG}")
    event_logger.event(
        "index_build",
        "started",
        input_dir=input_dir,
        output_dir=output_dir,
        schema_guided=schema_guided,
        use_umls=use_umls,
        batch_size=batch_size,
        max_retries=max_retries,
        llm_provider=os.environ.get("INDEX_LLM_PROVIDER", os.environ.get("GENERATION_MODEL_PROVIDER", "ollama")),
        llm_model=os.environ.get("INDEX_LLM_MODEL", os.environ.get("GENERATOR_MODEL", "")),
        llm_request_timeout=float(os.environ.get("INDEX_LLM_REQUEST_TIMEOUT", str(DEFAULT_LLM_REQUEST_TIMEOUT))),
        embedding_provider=os.environ.get("INDEX_EMBEDDING_PROVIDER", os.environ.get("RAG_EMBEDDING_MODEL_PROVIDER", "ollama")),
        embedding_model=os.environ.get("INDEX_EMBEDDING_MODEL", os.environ.get("RAG_EMBEDDING_MODEL", "")),
        umls_enabled=env_bool("UMLS_ENABLED", False),
        umls_hint_limit=env_int("UMLS_HINT_LIMIT", DEFAULT_SCHEMA_HIT_LIMIT),
        schema_num_workers=int(os.environ.get("INDEX_SCHEMA_NUM_WORKERS", "1")),
        schema_max_triplets_per_chunk=int(os.environ.get("INDEX_SCHEMA_MAX_TRIPLETS_PER_CHUNK", "6")),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_source_documents(input_dir=input_dir)

    needs_rebuild, reasons = index_needs_rebuild(
        output_dir,
        input_dir,
        schema_guided=schema_guided,
        use_umls=use_umls,
    )
    if not needs_rebuild:
        progress_message(f"Using existing LlamaIndex property graph in {output_dir}")
        event_logger.event("index_build", "completed", reused_existing_index=True, output_dir=output_dir)
        return load_index(output_dir)

    progress_message("LlamaIndex index is missing or incompatible:")
    for reason in reasons:
        progress_message(f"  - {reason}")

    documents = source_documents(input_dir)
    if use_umls and not env_bool("UMLS_ENABLED", False):
        raise ValueError("UMLS_ENABLED must be true to build a UMLS-guided medical index.")

    guidance = None
    if use_umls:
        progress_message(f"Phase 1/3: building UMLS schema guidance for {len(documents)} source document(s)")
        guidance_started_at = time.perf_counter()
        guidance = build_schema_guidance(documents, input_dir, output_dir, event_logger=event_logger)
        progress_message(
            "Phase 1/3 complete: "
            f"{sum(guidance.concept_count_by_source.values())} concepts across "
            f"{len(guidance.concept_hints_by_source)} hinted document(s) in "
            f"{round(time.perf_counter() - guidance_started_at, 2)}s"
        )
    else:
        progress_message("Phase 1/3 skipped: UMLS guidance disabled")
    progress_message(
        f"Preparing index build for {len(documents)} source document(s) into {output_dir}"
    )
    event_logger.event(
        "schema_guidance",
        "completed",
        document_count=len(documents),
        umls_enabled=use_umls,
        candidate_term_count=(sum(guidance.candidate_count_by_source.values()) if guidance else 0),
        umls_concept_count=(sum(guidance.concept_count_by_source.values()) if guidance else 0),
        relation_hint_count=(guidance.relation_count if guidance else 0),
    )
    checkpoint_db = output_dir / CHECKPOINT_DB
    if checkpoint_db.exists():
        checkpoint_db.unlink()

    progress_message("Phase 2/3: initializing property graph index and validating Ollama models")
    index = _create_empty_index(schema_guided=schema_guided, validation_schema=(guidance.validation_schema if guidance else None))
    initial_counts = _graph_counts(index)
    if initial_counts is not None:
        progress_message(f"Phase 2/3 complete: initialized graph with {_format_graph_counts(initial_counts)}")
    conn = init_checkpoint_db(checkpoint_db)
    pending = documents

    if not pending:
        index.storage_context.persist(persist_dir=str(output_dir))
        manifest_path(output_dir).write_text(
            json.dumps(
                metadata_for_sources(
                    input_dir,
                    documents,
                    schema_guided=schema_guided,
                    use_umls=use_umls,
                    guidance=guidance,
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        return index

    batches = [pending[i : i + batch_size] for i in range(0, len(pending), batch_size)]
    last_error: Exception | None = None
    last_error_traceback: str | None = None
    total_documents = len(documents)
    processed_documents = 0

    for attempt in range(1, max_retries + 1):
        try:
            progress_message(
                f"Phase 3/3: indexing {total_documents} source document(s) "
                f"across {len(batches)} batch(es)"
            )
            progress_message(f"Index build attempt {attempt}/{max_retries}")
            for batch_number, batch in enumerate(
                iter_progress(
                    batches,
                    desc=f"Index batches (attempt {attempt})",
                    total=len(batches),
                    unit="batch",
                ),
                start=1,
            ):
                event_logger.event(
                    "index_batch",
                    "started",
                    attempt=attempt,
                    batch_size=len(batch),
                    first_source=batch[0].as_posix() if batch else None,
                    last_source=batch[-1].as_posix() if batch else None,
                )
                batch_docs = [
                    build_document(
                        path,
                        input_dir,
                        hint_block=(guidance.concept_hints_by_source.get(path.as_posix()) if guidance else None),
                    )
                    for path in batch
                ]
                progress_message(
                    f"Batch {batch_number}/{len(batches)}: inserting {len(batch_docs)} document(s)"
                )
                for doc in batch_docs:
                    mark_status(conn, doc.id_, "in_progress")
                for doc in iter_progress(
                    batch_docs,
                    desc=f"Docs in batch {batch_number}",
                    total=len(batch_docs),
                    unit="doc",
                ):
                    source_path = doc.metadata.get("source_path")
                    source_name = doc.metadata.get("source_name")
                    processed_documents += 1
                    hint_count = 0
                    concept_count = 0
                    candidate_count = 0
                    if guidance and source_path:
                        hint_count = 1 if guidance.concept_hints_by_source.get(source_path) else 0
                        candidate_count = guidance.candidate_count_by_source.get(source_path, 0)
                        concept_count = guidance.concept_count_by_source.get(source_path, 0)
                    doc_started_at = time.perf_counter()
                    graph_before = _graph_counts(index)
                    progress_message(
                        f"Indexing document {processed_documents}/{total_documents}: "
                        f"{source_name or doc.id_} "
                        f"(candidate_terms={candidate_count}, umls_concepts={concept_count})"
                    )
                    event_logger.event(
                        "index_document",
                        "started",
                        attempt=attempt,
                        source_path=source_path,
                        source_name=source_name,
                        document_id=doc.id_,
                        schema_guided=schema_guided,
                        umls_enabled=use_umls,
                        candidate_term_count=candidate_count,
                        umls_concept_count=concept_count,
                        has_concept_hint=bool(hint_count),
                    )
                    try:
                        index.insert(doc)
                    except Exception as exc:
                        event_logger.event(
                            "index_document",
                            "failed",
                            attempt=attempt,
                            source_path=source_path,
                            source_name=source_name,
                            document_id=doc.id_,
                            schema_guided=schema_guided,
                            umls_enabled=use_umls,
                            candidate_term_count=candidate_count,
                            umls_concept_count=concept_count,
                            has_concept_hint=bool(hint_count),
                            error_type=type(exc).__name__,
                            error=str(exc),
                            traceback=traceback.format_exc(limit=20),
                            duration_seconds=round(time.perf_counter() - doc_started_at, 2),
                        )
                        raise
                    graph_after = _graph_counts(index)
                    duration_seconds = round(time.perf_counter() - doc_started_at, 2)
                    progress_message(
                        f"Indexed document {processed_documents}/{total_documents}: "
                        f"{source_name or doc.id_} in {duration_seconds}s"
                        f"{_format_graph_delta(graph_before, graph_after)}"
                    )
                    event_logger.event(
                        "index_document",
                        "completed",
                        attempt=attempt,
                        source_path=source_path,
                        source_name=source_name,
                        document_id=doc.id_,
                        schema_guided=schema_guided,
                        umls_enabled=use_umls,
                        candidate_term_count=candidate_count,
                        umls_concept_count=concept_count,
                        has_concept_hint=bool(hint_count),
                        duration_seconds=duration_seconds,
                        graph_nodes=(graph_after.total_nodes if graph_after else None),
                        graph_entity_nodes=(graph_after.entity_nodes if graph_after else None),
                        graph_chunk_nodes=(graph_after.chunk_nodes if graph_after else None),
                        graph_relationships=(graph_after.relation_count if graph_after else None),
                        graph_triplets=(graph_after.triplet_count if graph_after else None),
                    )
                for doc in batch_docs:
                    mark_status(conn, doc.id_, "done")
                progress_message(f"Indexed batch of {len(batch_docs)} document(s)")
                event_logger.event(
                    "index_batch",
                    "completed",
                    attempt=attempt,
                    batch_size=len(batch_docs),
                    first_source=batch[0].as_posix() if batch else None,
                    last_source=batch[-1].as_posix() if batch else None,
                )

            index.storage_context.persist(persist_dir=str(output_dir))
            manifest_path(output_dir).write_text(
                json.dumps(
                    metadata_for_sources(
                        input_dir,
                        documents,
                        schema_guided=schema_guided,
                        use_umls=use_umls,
                        guidance=guidance,
                    ),
                    indent=2,
                ),
                encoding="utf-8",
            )
            event_logger.event(
                "index_build",
                "completed",
                output_dir=output_dir,
                document_count=len(documents),
                schema_guided=schema_guided,
                umls_enabled=use_umls,
            )
            progress_message(f"Index build completed: {output_dir}")
            return index
        except Exception as exc:
            last_error = exc
            last_error_traceback = traceback.format_exc(limit=20)
            error_text = str(exc).lower()
            non_retryable = isinstance(exc, (TypeError, ValueError)) or any(
                marker in error_text
                for marker in (
                    "no space left on device",
                    "not enough free disk space",
                    "max_model_len",
                    "unexpected keyword argument",
                )
            )
            if non_retryable:
                event_logger.event(
                    "index_build",
                    "failed",
                    attempt=attempt,
                    error_type=type(exc).__name__,
                    error=str(exc),
                    traceback=last_error_traceback,
                    retryable=False,
                )
                raise
            wait = 2**attempt
            event_logger.event(
                "index_build",
                "retrying",
                attempt=attempt,
                wait_seconds=wait,
                error_type=type(exc).__name__,
                error=str(exc),
                traceback=last_error_traceback,
            )
            progress_message(f"Attempt {attempt} failed: {exc}. Retrying in {wait}s...")
            time.sleep(wait)

    event_logger.event(
        "index_build",
        "failed",
        error_type=type(last_error).__name__ if last_error else None,
        error=str(last_error) if last_error else None,
        traceback=last_error_traceback,
        schema_guided=schema_guided,
        umls_enabled=use_umls,
    )
    raise last_error
