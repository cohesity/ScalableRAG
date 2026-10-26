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

"""SQLite-backed document and lexical stores for multi-million-doc corpora."""

from __future__ import annotations

import json
import re
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from .doc_store import DocMeta
from .lexical_index import LexicalSearchResult, _quote_fts


class PackedDocStore:
    """Read-only prefix view over a packed ``documents`` SQLite table."""

    def __init__(self, database: str | Path, limit: int):
        self.database = Path(database).resolve()
        self.limit = int(limit)
        with closing(self._connect()) as conn:
            available = int(conn.execute("SELECT count(*) FROM documents").fetchone()[0])
            if not 0 < self.limit <= available:
                raise ValueError(f"limit {self.limit} outside packed corpus of {available}")
            row = conn.execute(
                "SELECT bytes FROM corpus_sizes WHERE documents=?", (self.limit,)
            ).fetchone()
            self._total_bytes = int(row[0]) if row else int(conn.execute(
                "SELECT coalesce(sum(byte_count),0) FROM documents WHERE rowid<=?",
                (self.limit,),
            ).fetchone()[0])
            self._total_chars = int(conn.execute(
                "SELECT coalesce(sum(char_count),0) FROM documents WHERE rowid<=?",
                (self.limit,),
            ).fetchone()[0])
        self._all_ids_cache: list[str] | None = None
        # An ephemeral process-local read cache keeps repeated Zero-Ingestion
        # regex scans from benchmarking SQLite I/O instead of regex work.  It
        # is neither persisted nor indexed and its one-time build is timed.
        self._text_cache: list[str] | None = None
        self.text_cache_build_s = 0.0
        self._normalized_text_cache: list[str] | None = None
        self.normalized_text_cache_build_s = 0.0

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(f"file:{self.database}?mode=ro", uri=True)

    def all_doc_ids(self) -> list[str]:
        if self._all_ids_cache is None:
            self._all_ids_cache = [
                f"d{rowid:09d}" for rowid in range(1, self.limit + 1)
            ]
        return self._all_ids_cache

    def doc_count(self) -> int:
        return self.limit

    def get_text(self, doc_id: str) -> str:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT body FROM documents WHERE doc_id=? AND rowid<=?",
                (doc_id, self.limit),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown doc_id: {doc_id}")
        return str(row[0])

    @staticmethod
    def _meta(row: tuple) -> DocMeta:
        doc_id, filename, char_count, byte_count, source, title = row
        return DocMeta(
            doc_id=str(doc_id), filename=str(filename),
            char_count=int(char_count), byte_count=int(byte_count),
            word_count=0, estimated_tokens=max(1, int(char_count) * 10 // 35),
            extra={"source": str(source), "title": str(title)},
        )

    def get_meta(self, doc_id: str) -> DocMeta:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT doc_id,filename,char_count,byte_count,source,title "
                "FROM documents WHERE doc_id=? AND rowid<=?",
                (doc_id, self.limit),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown doc_id: {doc_id}")
        return self._meta(row)

    def resolve_doc_id(self, key: str) -> str | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT doc_id FROM documents WHERE rowid<=? AND "
                "(doc_id=? OR filename=? OR title=?) LIMIT 1",
                (self.limit, key, key, key),
            ).fetchone()
        return str(row[0]) if row else None

    def iter_docs(self) -> Iterator[tuple[str, DocMeta]]:
        with closing(self._connect()) as conn:
            for row in conn.execute(
                "SELECT doc_id,filename,char_count,byte_count,source,title "
                "FROM documents WHERE rowid<=? ORDER BY rowid", (self.limit,)
            ):
                meta = self._meta(row)
                yield meta.doc_id, meta

    def iter_texts(
        self, doc_ids: Iterable[str] | None = None
    ) -> Iterator[tuple[str, str]]:
        ids = (
            None if doc_ids is None
            else doc_ids if isinstance(doc_ids, list)
            else list(doc_ids)
        )
        if ids is None or len(ids) == self.limit:
            if self._text_cache is None:
                started = time.perf_counter()
                cache: list[str] = []
                with closing(self._connect()) as conn:
                    for (body,) in conn.execute(
                        "SELECT body FROM documents WHERE rowid<=? ORDER BY rowid",
                        (self.limit,),
                    ):
                        cache.append(str(body))
                self._text_cache = cache
                self.text_cache_build_s = time.perf_counter() - started
            for rowid, body in enumerate(self._text_cache, start=1):
                yield f"d{rowid:09d}", body
            return

        if self._text_cache is not None:
            for doc_id in ids:
                try:
                    rowid = int(doc_id[1:])
                except (TypeError, ValueError):
                    continue
                if 1 <= rowid <= self.limit:
                    yield doc_id, self._text_cache[rowid - 1]
            return

        with closing(self._connect()) as conn:
            for start in range(0, len(ids), 900):
                batch = ids[start:start + 900]
                marks = ",".join("?" for _ in batch)
                rows = conn.execute(
                    f"SELECT doc_id,body FROM documents WHERE doc_id IN ({marks})",
                    batch,
                ).fetchall()
                bodies = {str(doc_id): str(body) for doc_id, body in rows}
                for doc_id in batch:
                    if doc_id in bodies:
                        yield doc_id, bodies[doc_id]

    def total_bytes(self) -> int:
        return self._total_bytes

    def total_chars(self) -> int:
        return self._total_chars

    def total_estimated_tokens(self) -> int:
        return max(1, self.total_chars() * 10 // 35)

    def iter_normalized_texts(
        self, doc_ids: Iterable[str] | None = None
    ) -> Iterator[tuple[str, str]]:
        """Yield process-local normalized text for repeated exact scans."""
        from ..retrieval.text_normalization import normalize_for_regex

        ids = (
            None if doc_ids is None
            else doc_ids if isinstance(doc_ids, list)
            else list(doc_ids)
        )
        if ids is None or len(ids) == self.limit:
            if self._normalized_text_cache is None:
                started = time.perf_counter()
                self._normalized_text_cache = [
                    normalize_for_regex(text) for _, text in self.iter_texts()
                ]
                self.normalized_text_cache_build_s = time.perf_counter() - started
            for rowid, text in enumerate(self._normalized_text_cache, start=1):
                yield f"d{rowid:09d}", text
            return

        if self._normalized_text_cache is not None:
            for doc_id in ids:
                try:
                    rowid = int(doc_id[1:])
                except (TypeError, ValueError):
                    continue
                if 1 <= rowid <= self.limit:
                    yield doc_id, self._normalized_text_cache[rowid - 1]
            return

        for doc_id, text in self.iter_texts(ids):
            yield doc_id, normalize_for_regex(text)

    def keyword_landscape(
        self, tokens: Sequence[str], *, materialize_limit: int = 16_384
    ) -> dict[str, tuple[int, str | None, set[str]]]:
        """Count several substrings in one sequential corpus pass."""
        ordered = list(dict.fromkeys(token for token in tokens if token))
        lowered = [token.casefold() for token in ordered]
        counts = [0] * len(ordered)
        first: list[str | None] = [None] * len(ordered)
        ids: list[set[str]] = [set() for _ in ordered]
        for doc_id, text in self.iter_normalized_texts():
            haystack = text.casefold()
            for index, token in enumerate(lowered):
                if token in haystack:
                    counts[index] += 1
                    if first[index] is None:
                        first[index] = doc_id
                    if counts[index] <= materialize_limit:
                        ids[index].add(doc_id)
                    elif counts[index] == materialize_limit + 1:
                        ids[index].clear()
        return {
            token: (counts[index], first[index], ids[index])
            for index, token in enumerate(ordered)
        }

    def set_byte_size(self, doc_ids: list[str] | set[str]) -> int:
        if len(doc_ids) == self.limit:
            return self._total_bytes
        ids = list(doc_ids)
        total = 0
        with closing(self._connect()) as conn:
            for start in range(0, len(ids), 900):
                batch = ids[start:start + 900]
                marks = ",".join("?" for _ in batch)
                total += int(conn.execute(
                    f"SELECT coalesce(sum(byte_count),0) FROM documents "
                    f"WHERE doc_id IN ({marks})", batch,
                ).fetchone()[0])
        return total

    def count_filename_matches(
        self, pattern: str, doc_ids: Iterable[str]
    ) -> int:
        """Count regex filename matches without one connection per document."""
        compiled = re.compile(pattern, re.IGNORECASE)
        ids = doc_ids if isinstance(doc_ids, list) else list(doc_ids)
        count = 0
        with closing(self._connect()) as conn:
            if len(ids) == self.limit:
                rows = conn.execute(
                    "SELECT filename FROM documents WHERE rowid<=?",
                    (self.limit,),
                )
                return sum(1 for (filename,) in rows if compiled.search(str(filename)))
            for start in range(0, len(ids), 900):
                batch = ids[start:start + 900]
                marks = ",".join("?" for _ in batch)
                rows = conn.execute(
                    f"SELECT filename FROM documents WHERE doc_id IN ({marks})",
                    batch,
                )
                count += sum(
                    1 for (filename,) in rows if compiled.search(str(filename))
                )
        return count

    def filename_landscape(self, tokens: Sequence[str]) -> dict[str, int]:
        """Count several filename substrings in one metadata-table pass."""
        ordered = list(dict.fromkeys(token for token in tokens if token))
        lowered = [token.casefold() for token in ordered]
        counts = [0] * len(ordered)
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT filename FROM documents WHERE rowid<=?", (self.limit,)
            )
            for (filename,) in rows:
                haystack = str(filename).casefold()
                for index, token in enumerate(lowered):
                    if token in haystack:
                        counts[index] += 1
        return {token: counts[index] for index, token in enumerate(ordered)}


class PackedLexicalIndex:
    """Physical FTS5 index for one exact packed-corpus scale point."""

    def __init__(self, database: str | Path, limit: int):
        self.path = Path(database).resolve()
        self.limit = int(limit)
        self._fts_table: str | None = None
        if self.path.is_file():
            with closing(self._connect()) as conn:
                total_row = conn.execute(
                    "SELECT max(documents) FROM corpus_sizes"
                ).fetchone()
                total = int(total_row[0]) if total_row and total_row[0] else 0
                candidate = (
                    "documents_fts" if self.limit == total
                    else f"documents_fts_{self.limit}"
                )
                exists = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                    (candidate,),
                ).fetchone()
                if exists:
                    self._fts_table = candidate

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)

    def exists(self) -> bool:
        return self._fts_table is not None

    def validate(self, doc_store: PackedDocStore) -> None:
        if doc_store.database != self.path or doc_store.limit != self.limit:
            raise RuntimeError("Packed lexical index and document view differ")
        if self._fts_table is None:
            raise RuntimeError(
                f"No physical packed lexical index for {self.limit} documents. "
                "Build the exact prefix index before running Limited-Ingestion."
            )

    def document_count(self) -> int:
        return self.limit

    def metadata(self) -> dict[str, object]:
        with closing(self._connect()) as conn:
            rows = conn.execute("SELECT key,value FROM metadata").fetchall()
        data = {key: json.loads(value) for key, value in rows}
        data["document_count"] = self.limit
        data["physical_fts_table"] = self._fts_table
        return data

    @property
    def _table(self) -> str:
        if self._fts_table is None:
            raise RuntimeError(
                f"No physical packed lexical index for {self.limit} documents"
            )
        return self._fts_table

    @staticmethod
    def _expression(terms: Sequence[str]) -> str:
        cleaned = [term.strip() for term in terms if term and term.strip()]
        if not cleaned:
            raise ValueError("At least one non-empty keyword is required")
        return " OR ".join(_quote_fts(term) for term in cleaned)

    def search_any(
        self, terms: Sequence[str], *, target_ids: Iterable[str] | None = None,
        limit: int = 16_384,
    ) -> LexicalSearchResult:
        expression = self._expression(terms)
        target = (
            None if target_ids is None
            else target_ids if isinstance(target_ids, list)
            else list(target_ids)
        )
        table = self._table
        with closing(self._connect()) as conn:
            if target is None or len(target) == self.limit:
                rows = conn.execute(
                    f"SELECT d.doc_id FROM {table} f JOIN documents d "
                    f"ON d.rowid=f.rowid WHERE {table} MATCH ? LIMIT ?",
                    (expression, limit + 1),
                ).fetchall()
            else:
                conn.execute("CREATE TEMP TABLE scope(doc_id TEXT PRIMARY KEY)")
                conn.executemany("INSERT OR IGNORE INTO scope VALUES (?)", ((x,) for x in target))
                rows = conn.execute(
                    f"SELECT d.doc_id FROM {table} f JOIN documents d ON d.rowid=f.rowid "
                    f"JOIN scope s ON s.doc_id=d.doc_id WHERE {table} MATCH ? LIMIT ?",
                    (expression, limit + 1),
                ).fetchall()
        return LexicalSearchResult(
            [str(row[0]) for row in rows[:limit]], len(rows) > limit
        )

    def count_any(
        self, terms: Sequence[str], *, target_ids: Iterable[str] | None = None
    ) -> int:
        expression = self._expression(terms)
        target = (
            None if target_ids is None
            else target_ids if isinstance(target_ids, list)
            else list(target_ids)
        )
        table = self._table
        with closing(self._connect()) as conn:
            if target is None or len(target) == self.limit:
                row = conn.execute(
                    f"SELECT count(*) FROM {table} WHERE {table} MATCH ?",
                    (expression,),
                ).fetchone()
            else:
                conn.execute("CREATE TEMP TABLE scope(doc_id TEXT PRIMARY KEY)")
                conn.executemany("INSERT OR IGNORE INTO scope VALUES (?)", ((x,) for x in target))
                row = conn.execute(
                    f"SELECT count(*) FROM {table} f JOIN documents d ON d.rowid=f.rowid "
                    f"JOIN scope s ON s.doc_id=d.doc_id WHERE {table} MATCH ?",
                    (expression,),
                ).fetchone()
        return int(row[0] if row else 0)

    def search_filenames(self, terms: Sequence[str], *, limit: int = 16_384) -> LexicalSearchResult:
        """Use the indexed title column as filename candidates."""
        expression = "title : (" + self._expression(terms) + ")"
        table = self._table
        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"SELECT d.doc_id FROM {table} f JOIN documents d ON d.rowid=f.rowid "
                f"WHERE {table} MATCH ? LIMIT ?", (expression, limit + 1),
            ).fetchall()
        return LexicalSearchResult([str(r[0]) for r in rows[:limit]], len(rows) > limit)

    def count_filenames(self, term: str) -> int:
        expression = "title : (" + self._expression([term]) + ")"
        table = self._table
        with closing(self._connect()) as conn:
            row = conn.execute(
                f"SELECT count(*) FROM {table} WHERE {table} MATCH ?", (expression,),
            ).fetchone()
        return int(row[0] if row else 0)
