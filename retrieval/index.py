import hashlib
import json
import os
import sqlite3
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from helpers.config import env_bool, env_int
from helpers.jsonl import JsonlLogger, new_run_id
from helpers.ollama import assert_ollama_available, ollama_endpoint
from helpers.paths import INDEX_BUILD_LOG_PATH
from helpers.progress import iter_progress, progress_message
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
from retrieval.concepts.umls import UMLSClient, UMLSConfig


INDEX_MANIFEST = "index_manifest.json"
CHECKPOINT_DB = "index_checkpoints.sqlite"
SOURCE_FILE_PATTERN = "*.txt"
DEFAULT_BATCH_SIZE = 20
DEFAULT_MAX_RETRIES = 3
DEFAULT_SCHEMA_HIT_LIMIT = 120
INDEX_EVENT_LOG = INDEX_BUILD_LOG_PATH
DEFAULT_LLM_REQUEST_TIMEOUT = 180.0


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
        "You can run `python make_index.py extract-mimic-discharge --csv-path data/mimic_iv_note/discharge.csv`."
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


def _umls_client() -> UMLSClient | None:
    if not env_bool("UMLS_ENABLED", False):
        return None
    return UMLSClient(UMLSConfig.from_env())


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


def build_schema_guidance(documents: list[Path], input_dir: Path) -> ClinicalSchemaGuidance:
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

    extractor = UMLSConceptExtractor(client)
    hint_by_source: dict[str, str] = {}
    counts_by_source: dict[str, int] = {}
    candidate_counts_by_source: dict[str, int] = {}
    observed_entity_types: set[str] = set()

    limit = env_int("UMLS_HINT_LIMIT", DEFAULT_SCHEMA_HIT_LIMIT)
    for path in documents:
        text = read_source_file(path)
        candidate_terms = extract_candidate_terms(text, limit=limit)
        candidate_counts_by_source[path.as_posix()] = len(candidate_terms)
        mentions = _dedupe_mentions(extractor.extract_from_terms(text, candidate_terms))
        counts_by_source[path.as_posix()] = len(mentions)
        observed_entity_types.update(
            {
                entity_type_for_category(mention.category or (mention.concept.category if mention.concept else None))
                for mention in mentions
            }
        )
        hint_block = format_concept_hint_block(mentions, max_mentions=20, max_relations=20)
        if hint_block:
            hint_by_source[path.as_posix()] = hint_block

    validation_schema = build_validation_schema({entity_type for entity_type in observed_entity_types if entity_type})
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
    assert_ollama_available(role="LLM", model_name=resolved_model)
    return Ollama(
        model=resolved_model,
        base_url=ollama_endpoint(),
        request_timeout=request_timeout,
    )


def build_embed_model():
    model_name = os.environ.get("INDEX_EMBEDDING_MODEL", os.environ.get("RAG_EMBEDDING_MODEL", "")).strip()
    from llama_index.embeddings.ollama import OllamaEmbedding

    resolved_model = model_name or "embeddinggemma:300m"
    assert_ollama_available(role="embedding", model_name=resolved_model)
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
        "umls_enabled": env_bool("UMLS_ENABLED", False),
        "umls_concept_count": sum((guidance.concept_count_by_source.get(path.as_posix(), 0) if guidance else 0) for path in documents),
        "candidate_term_count": sum((guidance.candidate_count_by_source.get(path.as_posix(), 0) if guidance else 0) for path in documents),
        "relation_hint_count": guidance.relation_count if guidance else 0,
        "updated_at": time.time(),
    }


def index_needs_rebuild(output_dir: Path, input_dir: Path, schema_guided: bool) -> tuple[bool, list[str]]:
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
    if manifest.get("llm_provider") != current_llm_provider or manifest.get("llm_model") != current_llm_model:
        reasons.append("LLM model changed")
    if manifest.get("embedding_provider") != current_embedding_provider or manifest.get("embedding_model") != current_embedding_model:
        reasons.append("embedding model changed")

    current_fingerprint = fingerprint_documents(documents, input_dir)
    if manifest.get("source_fingerprint") != current_fingerprint:
        reasons.append("source fingerprint changed")

    return bool(reasons), reasons


def _load_existing_index(output_dir: Path):
    from llama_index.core import StorageContext, load_index_from_storage

    storage_context = StorageContext.from_defaults(persist_dir=str(output_dir))
    try:
        index = load_index_from_storage(storage_context)
    except Exception:
        from llama_index.core.indices.property_graph import PropertyGraphIndex

        if hasattr(PropertyGraphIndex, "from_existing"):
            index = PropertyGraphIndex.from_existing(
                property_graph_store=storage_context.property_graph_store,
                vector_store=getattr(storage_context, "vector_store", None),
                llm=build_llm(),
                embed_model=build_embed_model(),
            )
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

    llm = build_llm()
    embed_model = build_embed_model()
    if schema_guided:
        schema_num_workers = int(os.environ.get("INDEX_SCHEMA_NUM_WORKERS", "1"))
        schema_triplets = int(os.environ.get("INDEX_SCHEMA_MAX_TRIPLETS_PER_CHUNK", "6"))
        kg_extractor = SchemaLLMPathExtractor(
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

    needs_rebuild, reasons = index_needs_rebuild(output_dir, input_dir, schema_guided=schema_guided)
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

    guidance = build_schema_guidance(documents, input_dir) if use_umls else None
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

    index = _create_empty_index(schema_guided=schema_guided, validation_schema=(guidance.validation_schema if guidance else None))
    conn = init_checkpoint_db(checkpoint_db)
    pending = documents

    if not pending:
        index.storage_context.persist(persist_dir=str(output_dir))
        manifest_path(output_dir).write_text(
            json.dumps(metadata_for_sources(input_dir, documents, schema_guided=schema_guided, guidance=guidance), indent=2),
            encoding="utf-8",
        )
        return index

    batches = [pending[i : i + batch_size] for i in range(0, len(pending), batch_size)]
    last_error: Exception | None = None
    last_error_traceback: str | None = None

    for attempt in range(1, max_retries + 1):
        try:
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
                    hint_count = 0
                    concept_count = 0
                    candidate_count = 0
                    if guidance and source_path:
                        hint_count = 1 if guidance.concept_hints_by_source.get(source_path) else 0
                        candidate_count = guidance.candidate_count_by_source.get(source_path, 0)
                        concept_count = guidance.concept_count_by_source.get(source_path, 0)
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
                        )
                        raise
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
                json.dumps(metadata_for_sources(input_dir, documents, schema_guided=schema_guided, guidance=guidance), indent=2),
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
