from __future__ import annotations

import argparse
import json
import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from helpers.jsonl import to_jsonable
from helpers.paths import CACHE_ROOT
from retrieval.concepts.schema import UMLSConcept
from retrieval.concepts.vocabularies import SOURCE_PRIORITY, category_for

DEFAULT_LOCAL_UMLS_DB_PATH = CACHE_ROOT / "umls_local.sqlite3"
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
    def __init__(self, db_path: Path, source_vocabularies: tuple[str, ...] = SOURCE_PRIORITY):
        self.db_path = Path(db_path)
        self.source_vocabularies = source_vocabularies
        self._search_cache: dict[tuple[str, tuple[str, ...] | None, int | None, str], list[UMLSConcept]] = {}
        self._best_match_cache: dict[tuple[str, tuple[str, ...] | None], UMLSConcept | None] = {}
        self._local = threading.local()
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Missing local UMLS database: {self.db_path}. "
                "Build it first with `python -m retrieval.concepts.local_umls build`."
            )

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

    def search(
        self,
        term: str,
        source_vocabularies: tuple[str, ...] | None = None,
        page_size: int | None = None,
        search_type: str = "words",
    ) -> list[UMLSConcept]:
        cache_key = (term.strip().lower(), source_vocabularies, page_size, search_type)
        if cache_key in self._search_cache:
            return self._search_cache[cache_key]

        normalized = normalize_umls_term(term)
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
        cache_key = (term.strip().lower(), source_vocabularies)
        if cache_key in self._best_match_cache:
            return self._best_match_cache[cache_key]

        matches = self.search(term, source_vocabularies=source_vocabularies, page_size=1)
        best = matches[0] if matches else None
        self._best_match_cache[cache_key] = best
        return best


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
            semantic_rows.append((cui, semantic_type, category_for(None, semantic_type)))
            if len(semantic_rows) >= config.batch_size:
                conn.executemany(
                    "INSERT INTO umls_semantic_types(cui, semantic_type, category) VALUES (?, ?, ?)",
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
                        INSERT OR REPLACE INTO umls_preferred_terms(cui, source_vocabulary, preferred_term)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or query a local UMLS SQLite index.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build a local SQLite lookup DB from UMLS META files.")
    build.add_argument("--meta-dir", default=str(DEFAULT_LOCAL_UMLS_META_DIR))
    build.add_argument("--db-path", default=str(DEFAULT_LOCAL_UMLS_DB_PATH))
    build.add_argument("--sources", default=",".join(SOURCE_PRIORITY))
    build.add_argument("--languages", default="ENG")
    build.add_argument("--batch-size", type=int, default=5000)

    lookup = subparsers.add_parser("lookup", help="Query a built local UMLS SQLite index.")
    lookup.add_argument("terms", nargs="+")
    lookup.add_argument("--db-path", default=str(DEFAULT_LOCAL_UMLS_DB_PATH))
    lookup.add_argument("--sources", default=",".join(SOURCE_PRIORITY))

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build":
        db_path = build_local_umls_database(
            LocalUMLSBuildConfig(
                meta_dir=Path(args.meta_dir),
                db_path=Path(args.db_path),
                source_vocabularies=tuple(token.strip() for token in args.sources.split(",") if token.strip()),
                languages=tuple(token.strip() for token in args.languages.split(",") if token.strip()),
                batch_size=args.batch_size,
            )
        )
        print(f"Built local UMLS database at {db_path}")
        return

    client = LocalUMLSClient(
        db_path=Path(args.db_path),
        source_vocabularies=tuple(token.strip() for token in args.sources.split(",") if token.strip()),
    )
    results = {term: client.search(term) for term in args.terms}
    print(json.dumps(to_jsonable(results), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
