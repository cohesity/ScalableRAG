#    Copyright (c) Cohesity 2026. All rights reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
 
#        http://www.apache.org/licenses/LICENSE-2.0
 
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""Portable full-text index used by Limited-Ingestion ScalableRAG.

The implementation uses SQLite FTS5's trigram tokenizer. SQLite is embedded,
open source, available through Python's standard library, and requires no
external service. Trigrams give case-insensitive substring candidate search,
which is a closer fit for literal regex branches than token-only retrieval.
Exact regex semantics are still provided by the retrieval layer after a set
has become small enough to scan safely.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .doc_store import DocStore


class LexicalIndexError(RuntimeError):
    """Raised when the local lexical index is unavailable or incompatible."""


@dataclass(frozen=True)
class LexicalSearchResult:
    doc_ids: list[str]
    truncated: bool


def _quote_fts(value: str) -> str:
    """Quote a literal for an FTS5 MATCH expression."""
    return '"' + value.replace('"', '""') + '"'


class LexicalIndex:
    """A document-level, case-insensitive SQLite FTS5 trigram index."""

    FILENAME = "lexical.sqlite3"
    SCHEMA_VERSION = 1

    def __init__(self, path: str | Path):
        self.path = Path(path)

    @classmethod
    def for_store(cls, store_dir: str | Path) -> "LexicalIndex":
        return cls(Path(store_dir) / cls.FILENAME)

    @staticmethod
    def corpus_fingerprint(doc_store: DocStore) -> str:
        digest = hashlib.sha256()
        for doc_id, meta in doc_store.iter_docs():
            digest.update(doc_id.encode("utf-8"))
            digest.update(b"\0")
            digest.update(meta.filename.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(meta.byte_count or meta.char_count).encode("ascii"))
            digest.update(b"\n")
        return digest.hexdigest()

    @classmethod
    def build(cls, store_dir: str | Path, doc_store: DocStore) -> "LexicalIndex":
        """Build an index atomically from the original document texts."""
        store = Path(store_dir)
        store.mkdir(parents=True, exist_ok=True)
        final_path = store / cls.FILENAME
        fd, tmp_name = tempfile.mkstemp(
            prefix="lexical.", suffix=".sqlite3.tmp", dir=store
        )
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            conn = sqlite3.connect(tmp_path)
            try:
                conn.execute("PRAGMA journal_mode=OFF")
                conn.execute("PRAGMA synchronous=OFF")
                try:
                    conn.execute(
                        "CREATE VIRTUAL TABLE documents USING fts5("
                        "doc_id UNINDEXED, body, tokenize='trigram')"
                    )
                except sqlite3.OperationalError as exc:
                    raise LexicalIndexError(
                        "This Python/SQLite build does not provide the FTS5 "
                        "trigram tokenizer. Install a standard SQLite build "
                        "with FTS5 enabled."
                    ) from exc
                conn.execute(
                    "CREATE TABLE index_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
                rows = (
                    (doc_id, doc_store.get_text(doc_id))
                    for doc_id in doc_store.all_doc_ids()
                )
                conn.executemany(
                    "INSERT INTO documents(doc_id, body) VALUES (?, ?)", rows
                )
                metadata = {
                    "schema_version": cls.SCHEMA_VERSION,
                    "document_count": doc_store.doc_count(),
                    "total_bytes": doc_store.total_bytes(),
                    "corpus_fingerprint": cls.corpus_fingerprint(doc_store),
                    "tokenizer": "trigram",
                }
                conn.executemany(
                    "INSERT INTO index_metadata(key, value) VALUES (?, ?)",
                    [(key, json.dumps(value)) for key, value in metadata.items()],
                )
                conn.commit()
            finally:
                conn.close()
            os.replace(tmp_path, final_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        return cls(final_path)

    def exists(self) -> bool:
        return self.path.is_file()

    def _connect(self) -> sqlite3.Connection:
        if not self.exists():
            raise LexicalIndexError(f"Lexical index not found: {self.path}")
        return sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)

    def metadata(self) -> dict[str, object]:
        with closing(self._connect()) as conn:
            try:
                rows = conn.execute("SELECT key, value FROM index_metadata").fetchall()
            except sqlite3.DatabaseError as exc:
                raise LexicalIndexError(f"Invalid lexical index: {self.path}") from exc
        return {key: json.loads(value) for key, value in rows}

    def validate(self, doc_store: DocStore) -> None:
        meta = self.metadata()
        if meta.get("schema_version") != self.SCHEMA_VERSION:
            raise LexicalIndexError("Lexical index schema is out of date; rebuild it.")
        expected = self.corpus_fingerprint(doc_store)
        if meta.get("corpus_fingerprint") != expected:
            raise LexicalIndexError(
                "Lexical index does not match the document store; rebuild ingestion."
            )

    def document_count(self) -> int:
        return int(self.metadata().get("document_count", 0))

    @staticmethod
    def _match_expression(terms: Sequence[str]) -> str:
        cleaned = [term.strip() for term in terms if term and term.strip()]
        if not cleaned:
            raise ValueError("At least one non-empty literal term is required")
        too_short = [term for term in cleaned if len(term) < 3]
        if too_short:
            raise LexicalIndexError(
                "Trigram lookup requires literals of at least three characters: "
                + ", ".join(repr(term) for term in too_short)
            )
        return " OR ".join(_quote_fts(term) for term in cleaned)

    def search_any(
        self,
        terms: Sequence[str],
        *,
        target_ids: Iterable[str] | None = None,
        limit: int = 16_384,
    ) -> LexicalSearchResult:
        """Return documents containing any literal, capped without truncation."""
        if limit <= 0:
            raise ValueError("limit must be positive")
        expr = self._match_expression(terms)
        target = None if target_ids is None else list(dict.fromkeys(target_ids))
        with closing(self._connect()) as conn:
            if target is None or len(target) == self.document_count():
                rows = conn.execute(
                    "SELECT doc_id FROM documents WHERE documents MATCH ? LIMIT ?",
                    (expr, limit + 1),
                ).fetchall()
            else:
                conn.execute("CREATE TEMP TABLE target_scope(doc_id TEXT PRIMARY KEY)")
                conn.executemany(
                    "INSERT OR IGNORE INTO target_scope(doc_id) VALUES (?)",
                    ((doc_id,) for doc_id in target),
                )
                rows = conn.execute(
                    "SELECT documents.doc_id FROM documents "
                    "JOIN target_scope ON target_scope.doc_id = documents.doc_id "
                    "WHERE documents MATCH ? LIMIT ?",
                    (expr, limit + 1),
                ).fetchall()
        truncated = len(rows) > limit
        return LexicalSearchResult(
            doc_ids=[str(row[0]) for row in rows[:limit]],
            truncated=truncated,
        )

    def count_any(
        self,
        terms: Sequence[str],
        *,
        target_ids: Iterable[str] | None = None,
    ) -> int:
        expr = self._match_expression(terms)
        target = None if target_ids is None else list(dict.fromkeys(target_ids))
        with closing(self._connect()) as conn:
            if target is None or len(target) == self.document_count():
                row = conn.execute(
                    "SELECT count(*) FROM documents WHERE documents MATCH ?", (expr,)
                ).fetchone()
            else:
                conn.execute("CREATE TEMP TABLE target_scope(doc_id TEXT PRIMARY KEY)")
                conn.executemany(
                    "INSERT OR IGNORE INTO target_scope(doc_id) VALUES (?)",
                    ((doc_id,) for doc_id in target),
                )
                row = conn.execute(
                    "SELECT count(*) FROM documents "
                    "JOIN target_scope ON target_scope.doc_id = documents.doc_id "
                    "WHERE documents MATCH ?",
                    (expr,),
                ).fetchone()
        return int(row[0] if row else 0)
