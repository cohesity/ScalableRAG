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

"""Tests for reusable cached pattern extractions."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scalablerag.auto_patterns.extraction_index import (
    ExtractionIndex,
    build_extraction_index,
)
from scalablerag.auto_patterns.patterns import PatternLibrary, PatternTemplate


class CountingStore:
    def __init__(self):
        self.texts = {
            "d1": "Program: Alpha\nStatus: APPROVED",
            "d2": "Program: Beta\nStatus: DENIED",
            "d3": "Unrelated document",
        }
        self.reads = 0

    def all_doc_ids(self):
        return list(self.texts)

    def get_text(self, doc_id):
        self.reads += 1
        return self.texts[doc_id]

    def get_meta(self, doc_id):
        return SimpleNamespace(filename=f"{doc_id}.txt")

    def iter_docs(self):
        return [
            (doc_id, SimpleNamespace(filename=f"{doc_id}.txt"))
            for doc_id in self.texts
        ]


class ExtractionIndexCacheTests(unittest.TestCase):
    def _build(self):
        library = PatternLibrary()
        library.add(PatternTemplate(
            name="program_status",
            description="Program and status fields",
            parameters=["program", "status"],
            regex_template=(
                r"Program:\s*(?P<program>[^\n]+)\n"
                r"Status:\s*(?P<status>[^\n]+)"
            ),
            validation_score=1.0,
        ))
        store = CountingStore()
        return build_extraction_index(library, store, verbose=False), store

    def test_search_uses_cached_captures_and_intersects_target(self):
        index, store = self._build()
        store.reads = 0
        results = index.search(
            "program_status", {"program": "Alpha"}, store,
            target_doc_ids=["d1", "d3"],
        )
        self.assertEqual([row["doc_id"] for row in results], ["d1"])
        self.assertEqual(store.reads, 0)

    def test_cached_captures_survive_round_trip(self):
        index, store = self._build()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "patterns.extraction.json"
            index.save(path)
            loaded = ExtractionIndex.load(path)
            store.reads = 0
            results = loaded.search(
                "program_status", {"status": "DENIED"}, store,
            )
        self.assertEqual([row["doc_id"] for row in results], ["d2"])
        self.assertEqual(store.reads, 0)


if __name__ == "__main__":
    unittest.main()
