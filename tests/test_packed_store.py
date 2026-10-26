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

"""Tests for packed nested-corpus storage and lexical lookup."""

from __future__ import annotations

import json
import sqlite3

from scalablerag.storage.packed_store import PackedDocStore, PackedLexicalIndex


def _database(path):
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE documents (
          rowid INTEGER PRIMARY KEY, doc_id TEXT, filename TEXT, title TEXT,
          source TEXT, byte_count INTEGER, char_count INTEGER,
          content_hash BLOB, body TEXT
        );
        CREATE TABLE corpus_sizes(documents INTEGER PRIMARY KEY, bytes INTEGER,
                                  wikipedia_documents INTEGER);
        CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT);
        """
    )
    bodies = ["Alpha evidence", "Beta distractor", "Alpha outside prefix"]
    for index, body in enumerate(bodies, 1):
        conn.execute(
            "INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?)",
            (index, f"d{index:09d}", f"{index}.txt", str(index), "wikipedia",
             len(body), len(body), bytes([index]), body),
        )
    conn.executemany(
        "INSERT INTO corpus_sizes VALUES (?,?,?)", [(2, 29, 2), (3, 49, 3)]
    )
    conn.execute("INSERT INTO metadata VALUES ('documents', ?)", (json.dumps(3),))
    conn.execute(
        "CREATE VIRTUAL TABLE documents_fts USING fts5(title,body,"
        "content='documents',content_rowid='rowid')"
    )
    conn.execute("INSERT INTO documents_fts(documents_fts) VALUES ('rebuild')")
    conn.execute(
        "CREATE VIRTUAL TABLE documents_fts_2 USING fts5("
        "title,body,content='',content_rowid='rowid')"
    )
    conn.execute(
        "INSERT INTO documents_fts_2(rowid,title,body) "
        "SELECT rowid,title,body FROM documents WHERE rowid<=2"
    )
    conn.commit()
    conn.close()


def test_packed_prefix_and_lexical_search(tmp_path):
    path = tmp_path / "corpus.sqlite3"
    _database(path)
    store = PackedDocStore(path, 2)
    index = PackedLexicalIndex(path, 2)
    assert store.all_doc_ids() == ["d000000001", "d000000002"]
    assert list(store.iter_texts())[-1][1] == "Beta distractor"
    assert index.count_any(["Alpha"]) == 1
    assert index.search_any(["Alpha"]).doc_ids == ["d000000001"]
    assert index.metadata()["physical_fts_table"] == "documents_fts_2"
    assert store.count_filename_matches(r"1|2", store.all_doc_ids()) == 2
    assert store.count_filename_matches(r"2", ["d000000002"]) == 1
    assert store.filename_landscape(["1", ".txt", "missing"]) == {
        "1": 1,
        ".txt": 2,
        "missing": 0,
    }


def test_packed_prefix_requires_a_physical_index(tmp_path):
    path = tmp_path / "corpus.sqlite3"
    _database(path)
    conn = sqlite3.connect(path)
    conn.execute("DROP TABLE documents_fts_2")
    conn.commit()
    conn.close()

    store = PackedDocStore(path, 2)
    index = PackedLexicalIndex(path, 2)
    assert not index.exists()
    try:
        index.validate(store)
    except RuntimeError as exc:
        assert "physical packed lexical index" in str(exc)
    else:
        raise AssertionError("missing physical prefix index was accepted")


def test_packed_full_scan_is_cached_in_memory(tmp_path):
    path = tmp_path / "corpus.sqlite3"
    _database(path)
    store = PackedDocStore(path, 3)

    first = list(store.iter_texts())
    path.rename(path.with_suffix(".moved"))
    second = list(store.iter_texts())

    assert second == first
    assert store.text_cache_build_s >= 0


def test_packed_normalized_scan_is_cached_in_memory(tmp_path):
    path = tmp_path / "corpus.sqlite3"
    _database(path)
    store = PackedDocStore(path, 3)

    first = list(store.iter_normalized_texts())
    path.rename(path.with_suffix(".moved"))
    second = list(store.iter_normalized_texts())

    assert second == first
    assert store.normalized_text_cache_build_s >= 0
