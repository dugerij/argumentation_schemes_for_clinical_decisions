import atexit
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from clinical_cds.io import to_jsonable
from clinical_cds.terminology.schema import UMLSConcept
from clinical_cds.terminology.vocabularies import SOURCE_PRIORITY, category_for

DEFAULT_LOCAL_UMLS_DB_PATH = Path("output/cache/umls_local.sqlite3")
DEFAULT_LOCAL_UMLS_META_DIR = Path("data/umls/META")
TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")


def normalize_umls_term(term: str) -> str:
    tokens = [token for token in TOKEN_SPLIT_RE.split(term.lower()) if token]
    return " ".join(tokens)


def resolve_umls_meta_dir(path: Path) -> Path:
    candidate = Path(path)
    if (candidate / "MRCONSO.RRF").exists() and (candidate / "MRSTY.RRF").exists():
        return candidate

    search_roots = [candidate]
    if not candidate.exists() and candidate.name == "META":
        search_roots.append(candidate.parent)

    if candidate.name != "META":
        direct_meta = candidate / "META"
        if (direct_meta / "MRCONSO.RRF").exists() and (direct_meta / "MRSTY.RRF").exists():
            return direct_meta

    matches: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        matches.extend(
            meta_dir
            for meta_dir in root.rglob("META")
            if (meta_dir / "MRCONSO.RRF").exists() and (meta_dir / "MRSTY.RRF").exists()
        )
    matches = sorted(set(matches))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Could not find a UMLS META directory under {candidate}. "
        "Expected MRCONSO.RRF and MRSTY.RRF."
    )


@dataclass(frozen=True)
class LocalUMLSBuildConfig:
    meta_dir: Path
    db_path: Path = DEFAULT_LOCAL_UMLS_DB_PATH
    source_vocabularies: tuple[str, ...] = SOURCE_PRIORITY
    languages: tuple[str, ...] = ("ENG",)
    batch_size: int = 5000


class LocalUMLSClient:
    def __init__(
        self,
        db_path: Path,
        source_vocabularies: tuple[str, ...] = SOURCE_PRIORITY,
        lookup_cache_db_path: Path | None = None,
        cache_commit_interval: int = 100,
    ):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Missing local UMLS database: {self.db_path}. "
                "Build it first with `python -m clinical_cds build-umls`."
            )
        database_stat = self.db_path.stat()
        database_identity = (
            f"{self.db_path.resolve()}|"
            f"{database_stat.st_size}|{database_stat.st_mtime_ns}"
        )
        self.database_id = hashlib.sha256(
            database_identity.encode("utf-8")
        ).hexdigest()[:16]
        self.source_vocabularies = source_vocabularies
        self._search_cache: dict[
            tuple[str, tuple[str, ...] | None, int | None, str],
            list[UMLSConcept],
        ] = {}
        self._best_match_cache: dict[
            tuple[str, tuple[str, ...] | None],
            UMLSConcept | None,
        ] = {}
        self._concept_terms_cache: dict[
            tuple[str, tuple[str, ...], int],
            tuple[str, ...],
        ] = {}
        self._supports_full_alias_lookup: bool | None = None
        self._local = threading.local()
        self._pending_cache_writes = 0
        self._cache_commit_interval = max(cache_commit_interval, 1)
        cache_setting = os.environ.get(
            "UMLS_LOCAL_LOOKUP_CACHE_ENABLED",
            "true",
        )
        self._lookup_cache_enabled = (
            cache_setting.strip().lower()
            not in {"0", "false", "no", "off"}
        )
        configured_lookup_cache = lookup_cache_db_path
        if configured_lookup_cache is None:
            configured_path = os.environ.get("UMLS_LOCAL_LOOKUP_CACHE_DB_PATH")
            configured_lookup_cache = (
                Path(configured_path)
                if configured_path
                else self.db_path.with_name(
                    f"{self.db_path.stem}_lookup_cache.sqlite3"
                )
        )
        self.lookup_cache_db_path = Path(configured_lookup_cache)
        if self._lookup_cache_enabled:
            self.lookup_cache_db_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.lookup_cache_db_path) as conn:
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute("PRAGMA synchronous = NORMAL")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS umls_lookup_cache (
                        normalized_term TEXT NOT NULL,
                        source_vocabularies TEXT NOT NULL,
                        concept_json TEXT,
                        updated_at REAL NOT NULL,
                        PRIMARY KEY (normalized_term, source_vocabularies)
                    )
                    """
                )
                conn.commit()
            atexit.register(self.flush_cache)

    def _get_connection(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.execute("PRAGMA query_only = ON")
            conn.execute("PRAGMA temp_store = MEMORY")
            conn.execute("PRAGMA mmap_size = 268435456")
            conn.execute("PRAGMA cache_size = -200000")
            self._local.conn = conn
        return conn

    def _get_lookup_cache_connection(self) -> sqlite3.Connection | None:
        if not self._lookup_cache_enabled:
            return None
        conn = getattr(self._local, "lookup_cache_conn", None)
        if conn is None:
            conn = sqlite3.connect(self.lookup_cache_db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            self._local.lookup_cache_conn = conn
        return conn

    def _sources_key(self, source_vocabularies: tuple[str, ...] | None) -> str:
        sources = source_vocabularies or self.source_vocabularies
        return f"{self.database_id}|{','.join(sources)}"

    def _cache_key(
        self,
        term: str,
        source_vocabularies: tuple[str, ...] | None,
    ) -> tuple[str, tuple[str, ...] | None]:
        normalized = normalize_umls_term(term)
        sources = source_vocabularies or self.source_vocabularies
        return normalized, tuple(sources) if sources else None

    def _load_persistent_best_match(
        self,
        normalized_term: str,
        source_vocabularies: tuple[str, ...] | None,
    ) -> tuple[bool, UMLSConcept | None]:
        conn = self._get_lookup_cache_connection()
        if conn is None or not normalized_term:
            return False, None

        row = conn.execute(
            """
            SELECT concept_json
            FROM umls_lookup_cache
            WHERE normalized_term = ? AND source_vocabularies = ?
            """,
            (normalized_term, self._sources_key(source_vocabularies)),
        ).fetchone()
        if row is None:
            return False, None
        concept_json = row[0]
        if not concept_json:
            return True, None
        payload = json.loads(concept_json)
        return True, UMLSConcept(
            cui=payload["cui"],
            preferred_term=payload["preferred_term"],
            semantic_type=payload["semantic_type"],
            source_vocabulary=payload.get("source_vocabulary", "UMLS"),
            source_code=payload.get("source_code"),
            category=payload.get("category"),
            aliases=tuple(payload.get("aliases", [])),
            metadata=payload.get("metadata", {}),
        )

    def _store_persistent_best_match(
        self,
        normalized_term: str,
        source_vocabularies: tuple[str, ...] | None,
        concept: UMLSConcept | None,
    ) -> None:
        conn = self._get_lookup_cache_connection()
        if conn is None or not normalized_term:
            return

        concept_json = None
        if concept is not None:
            concept_json = json.dumps(
                to_jsonable(
                    {
                        "cui": concept.cui,
                        "preferred_term": concept.preferred_term,
                        "semantic_type": concept.semantic_type,
                        "source_vocabulary": concept.source_vocabulary,
                        "source_code": concept.source_code,
                        "category": concept.category,
                        "aliases": list(concept.aliases),
                        "metadata": concept.metadata,
                    }
                ),
                ensure_ascii=False,
                sort_keys=True,
            )

        conn.execute(
            """
            INSERT INTO umls_lookup_cache(
                normalized_term,
                source_vocabularies,
                concept_json,
                updated_at
            )
            VALUES (?, ?, ?, ?)
            ON CONFLICT(normalized_term, source_vocabularies)
            DO UPDATE SET concept_json = excluded.concept_json,
                          updated_at = excluded.updated_at
            """,
            (normalized_term, self._sources_key(source_vocabularies), concept_json, time.time()),
        )
        self._pending_cache_writes += 1
        if self._pending_cache_writes >= self._cache_commit_interval:
            self.flush_cache()

    def flush_cache(self) -> None:
        conn = getattr(self._local, "lookup_cache_conn", None)
        if conn is None or self._pending_cache_writes == 0:
            return
        conn.commit()
        self._pending_cache_writes = 0

    @property
    def supports_full_alias_lookup(self) -> bool:
        if self._supports_full_alias_lookup is None:
            index_names = {
                str(row[1])
                for row in self._get_connection().execute(
                    "PRAGMA index_list('umls_terms')"
                )
            }
            self._supports_full_alias_lookup = (
                "idx_umls_terms_cui_source" in index_names
            )
        return self._supports_full_alias_lookup

    def search(
        self,
        term: str,
        source_vocabularies: tuple[str, ...] | None = None,
        page_size: int | None = None,
        search_type: str = "words",
    ) -> list[UMLSConcept]:
        normalized = normalize_umls_term(term)
        cache_key = (normalized, source_vocabularies, page_size, search_type)
        if cache_key in self._search_cache:
            return self._search_cache[cache_key]

        if not normalized:
            self._search_cache[cache_key] = []
            return []

        sources = source_vocabularies or self.source_vocabularies
        limit = page_size or 10

        query = """
            SELECT
                cui,
                preferred_term,
                semantic_type,
                source_vocabulary,
                source_code,
                category,
                term,
                is_preferred
            FROM umls_terms
            WHERE normalized_term = ?
        """
        params: list[object] = [normalized]
        if sources:
            placeholders = ",".join("?" for _ in sources)
            query += f" AND source_vocabulary IN ({placeholders})"
            params.extend(sources)
        query += """
            ORDER BY source_rank ASC, is_preferred DESC, length(term) DESC, term ASC
            LIMIT ?
        """
        params.append(limit)

        rows = self._get_connection().execute(query, params).fetchall()

        matches = [
            UMLSConcept(
                cui=row[0],
                preferred_term=row[1] or row[6],
                semantic_type=row[2] or "",
                source_vocabulary=row[3] or "UMLS",
                source_code=row[4],
                category=row[5],
                aliases=(row[6],) if row[6] and row[6] != row[1] else (),
                metadata={"backend": "local_umls"},
            )
            for row in rows
        ]
        self._search_cache[cache_key] = matches
        return matches

    def best_match(
        self,
        term: str,
        source_vocabularies: tuple[str, ...] | None = None,
    ) -> UMLSConcept | None:
        cache_key = self._cache_key(term, source_vocabularies)
        if cache_key in self._best_match_cache:
            return self._best_match_cache[cache_key]

        normalized_term = cache_key[0]
        if not normalized_term:
            self._best_match_cache[cache_key] = None
            return None

        found, cached = self._load_persistent_best_match(normalized_term, source_vocabularies)
        if found:
            self._best_match_cache[cache_key] = cached
            return cached

        matches = self.search(term, source_vocabularies=source_vocabularies, page_size=1)
        best = matches[0] if matches else None
        self._best_match_cache[cache_key] = best
        self._store_persistent_best_match(normalized_term, source_vocabularies, best)
        return best

    def concept_terms(
        self,
        cui: str,
        *,
        source_vocabularies: tuple[str, ...] | None = None,
        limit: int = 8,
    ) -> tuple[str, ...]:
        if limit < 1:
            return ()
        sources = tuple(source_vocabularies or self.source_vocabularies)
        cache_key = (cui, sources, limit)
        if cache_key in self._concept_terms_cache:
            return self._concept_terms_cache[cache_key]
        if self.supports_full_alias_lookup:
            table = "umls_terms"
            value_column = "term"
            order_clause = (
                "MIN(source_rank), MAX(is_preferred) DESC, "
                "length(term), term"
            )
        else:
            table = "umls_preferred_terms"
            value_column = "preferred_term"
            order_clause = "length(preferred_term), preferred_term"
        query = f"""
            SELECT {value_column}
            FROM {table}
            WHERE cui = ?
        """
        params: list[object] = [cui]
        if sources:
            placeholders = ",".join("?" for _ in sources)
            query += f" AND source_vocabulary IN ({placeholders})"
            params.extend(sources)
        group_column = (
            "normalized_term"
            if self.supports_full_alias_lookup
            else "preferred_term"
        )
        query += f"""
            GROUP BY {group_column}
            ORDER BY {order_clause}
            LIMIT ?
        """
        params.append(limit)
        rows = self._get_connection().execute(query, params).fetchall()
        terms = tuple(str(row[0]) for row in rows if row[0])
        self._concept_terms_cache[cache_key] = terms
        return terms


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;
        PRAGMA temp_store = MEMORY;
        DROP TABLE IF EXISTS umls_terms;
        DROP TABLE IF EXISTS umls_semantic_types;
        DROP TABLE IF EXISTS umls_preferred_terms;

        CREATE TABLE umls_semantic_types (
            cui TEXT PRIMARY KEY,
            semantic_type TEXT,
            category TEXT
        );

        CREATE TABLE umls_preferred_terms (
            cui TEXT NOT NULL,
            source_vocabulary TEXT NOT NULL,
            preferred_term TEXT NOT NULL,
            PRIMARY KEY (cui, source_vocabulary)
        );

        CREATE TABLE umls_terms (
            cui TEXT NOT NULL,
            normalized_term TEXT NOT NULL,
            term TEXT NOT NULL,
            preferred_term TEXT,
            semantic_type TEXT,
            source_vocabulary TEXT NOT NULL,
            source_code TEXT,
            category TEXT,
            is_preferred INTEGER NOT NULL,
            source_rank INTEGER NOT NULL
        );

        CREATE INDEX idx_umls_terms_normalized
            ON umls_terms (normalized_term);
        CREATE INDEX idx_umls_terms_normalized_source
            ON umls_terms (normalized_term, source_vocabulary, source_rank, is_preferred);
        CREATE INDEX idx_umls_terms_cui_source
            ON umls_terms (cui, source_vocabulary, source_rank, is_preferred);
        """
    )


def _iter_rrf_rows(path: Path) -> Iterable[list[str]]:
    with path.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            yield line.rstrip("\n").split("|")


def build_local_umls_database(config: LocalUMLSBuildConfig) -> Path:
    meta_dir = resolve_umls_meta_dir(config.meta_dir)
    mrconso_path = meta_dir / "MRCONSO.RRF"
    mrsty_path = meta_dir / "MRSTY.RRF"

    config.db_path.parent.mkdir(parents=True, exist_ok=True)
    source_rank = {source: index for index, source in enumerate(config.source_vocabularies)}
    language_set = set(config.languages)

    with sqlite3.connect(config.db_path) as conn:
        _ensure_schema(conn)

        semantic_rows: list[tuple[str, str, str | None]] = []
        seen_cuis: set[str] = set()
        for row in _iter_rrf_rows(mrsty_path):
            if len(row) < 4:
                continue
            cui = row[0].strip()
            semantic_type = row[3].strip()
            if not cui or cui in seen_cuis:
                continue
            seen_cuis.add(cui)
            semantic_rows.append(
                (cui, semantic_type, category_for(None, semantic_type))
            )
            if len(semantic_rows) >= config.batch_size:
                conn.executemany(
                    """
                    INSERT INTO umls_semantic_types(cui, semantic_type, category)
                    VALUES (?, ?, ?)
                    """,
                    semantic_rows,
                )
                conn.commit()
                semantic_rows.clear()
        if semantic_rows:
            conn.executemany(
                "INSERT INTO umls_semantic_types(cui, semantic_type, category) VALUES (?, ?, ?)",
                semantic_rows,
            )
            conn.commit()

        preferred_rows: list[tuple[str, str, str]] = []
        term_rows: list[tuple[str, str, str, str, str, int, int]] = []
        for row in _iter_rrf_rows(mrconso_path):
            if len(row) < 15:
                continue
            cui = row[0].strip()
            lat = row[1].strip()
            is_preferred = row[6].strip().upper() == "Y"
            source_vocabulary = row[11].strip()
            source_code = row[13].strip() or None
            term = row[14].strip()
            if not cui or not term:
                continue
            if lat not in language_set:
                continue
            if source_vocabulary not in source_rank:
                continue
            normalized_term = normalize_umls_term(term)
            if not normalized_term:
                continue
            if is_preferred:
                preferred_rows.append((cui, source_vocabulary, term))
            term_rows.append(
                (
                    cui,
                    normalized_term,
                    term,
                    source_vocabulary,
                    source_code,
                    int(is_preferred),
                    source_rank[source_vocabulary],
                )
            )
            if len(term_rows) >= config.batch_size:
                if preferred_rows:
                    conn.executemany(
                        """
                        INSERT OR REPLACE INTO umls_preferred_terms(
                            cui,
                            source_vocabulary,
                            preferred_term
                        )
                        VALUES (?, ?, ?)
                        """,
                        preferred_rows,
                    )
                    preferred_rows.clear()
                conn.executemany(
                    """
                    INSERT INTO umls_terms(
                        cui,
                        normalized_term,
                        term,
                        source_vocabulary,
                        source_code,
                        is_preferred,
                        source_rank
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    term_rows,
                )
                conn.commit()
                term_rows.clear()
        if preferred_rows:
            conn.executemany(
                """
                INSERT OR REPLACE INTO umls_preferred_terms(cui, source_vocabulary, preferred_term)
                VALUES (?, ?, ?)
                """,
                preferred_rows,
            )
        if term_rows:
            conn.executemany(
                """
                INSERT INTO umls_terms(
                    cui,
                    normalized_term,
                    term,
                    source_vocabulary,
                    source_code,
                    is_preferred,
                    source_rank
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                term_rows,
            )
        conn.commit()

        conn.execute(
            """
            UPDATE umls_terms
            SET preferred_term = COALESCE(
                (
                    SELECT p.preferred_term
                    FROM umls_preferred_terms p
                    WHERE p.cui = umls_terms.cui
                      AND p.source_vocabulary = umls_terms.source_vocabulary
                ),
                term
            )
            """
        )
        conn.execute(
            """
            UPDATE umls_terms
            SET semantic_type = (
                    SELECT s.semantic_type
                    FROM umls_semantic_types s
                    WHERE s.cui = umls_terms.cui
                ),
                category = COALESCE(
                    (
                        SELECT s.category
                        FROM umls_semantic_types s
                        WHERE s.cui = umls_terms.cui
                    ),
                    category
                )
            """
        )
        conn.commit()
        conn.execute("ANALYZE")
        conn.commit()

    return config.db_path
