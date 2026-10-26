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

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scalablerag.llm.base import BaseLLM
from scalablerag.retrieval.tiered_filter import (
    FilterBudgetExceeded,
    TieredFilterConfig,
    TieredRegexFilter,
    literal_alternation_terms,
)
from scalablerag.retrieval.tools import (
    ToolContext,
    execute_tool,
    tool_explore_patterns,
    tool_extract_field,
    tool_extract_list,
)
from scalablerag.storage.doc_store import DocStore
from scalablerag.storage.lexical_index import LexicalIndex, LexicalIndexError


class _DummyLLM(BaseLLM):
    def complete(self, prompt: str, max_tokens: int = 1024, *, json_schema=None) -> str:
        return ""


def _store(root: Path) -> DocStore:
    store = DocStore(root / "docs")
    store.add_document("d1", "Alpha project. Revenue: 10.", filename="one.txt")
    store.add_document("d2", "ALPHA team. Expenses: 5.", filename="two.txt")
    store.add_document("d3", "Beta project only.", filename="three.txt")
    store.add_document("d4", "Nothing relevant here.", filename="four.txt")
    return store


class LexicalIndexTests(unittest.TestCase):
    def test_case_insensitive_substring_search_and_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = _store(root)
            index = LexicalIndex.build(root, store)
            result = index.search_any(["alpha"])
            self.assertFalse(result.truncated)
            self.assertEqual(set(result.doc_ids), {"d1", "d2"})

            scoped = index.search_any(["alpha"], target_ids=["d2", "d3"])
            self.assertEqual(scoped.doc_ids, ["d2"])

    def test_index_detects_stale_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = _store(root)
            index = LexicalIndex.build(root, store)
            store.add_document("d5", "new document", filename="five.txt")
            with self.assertRaises(LexicalIndexError):
                index.validate(store)


class TieredRegexFilterTests(unittest.TestCase):
    def test_literal_parser(self):
        self.assertEqual(literal_alternation_terms("alpha|beta"), ["alpha", "beta"])
        self.assertIsNone(literal_alternation_terms("alpha.*beta"))
        self.assertIsNone(literal_alternation_terms("(alpha|beta)"))

    def test_large_literal_uses_index_then_exact_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = _store(root)
            index = LexicalIndex.build(root, store)
            policy = TieredRegexFilter(
                store,
                index,
                TieredFilterConfig(max_regex_bytes=10_000, max_regex_docs=2),
            )
            result = policy.apply("alpha", store.all_doc_ids())
            self.assertEqual(result.mode, "indexed_literal_confirmed")
            self.assertTrue(result.exact)
            self.assertEqual(set(result.positive_ids), {"d1", "d2"})

    def test_large_complex_regex_is_refused_without_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = _store(root)
            index = LexicalIndex.build(root, store)
            policy = TieredRegexFilter(
                store,
                index,
                TieredFilterConfig(max_regex_bytes=10_000, max_regex_docs=2),
            )
            with self.assertRaises(FilterBudgetExceeded):
                policy.apply(r"Alpha.*Revenue", store.all_doc_ids())

    def test_complex_regex_is_exact_after_narrowing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = _store(root)
            index = LexicalIndex.build(root, store)
            policy = TieredRegexFilter(
                store,
                index,
                TieredFilterConfig(max_regex_bytes=10_000, max_regex_docs=2),
            )
            result = policy.apply(r"Alpha.*Revenue", ["d1", "d2"])
            self.assertEqual(result.mode, "exact_regex")
            self.assertEqual(result.positive_ids, ["d1"])

    def test_tool_reports_indexed_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = _store(root)
            index = LexicalIndex.build(root, store)
            ctx = ToolContext(
                llm=_DummyLLM(),
                doc_store=store,
                embedding_store=None,
                embedder=None,
                use_embeddings=False,
                lexical_index=index,
                use_tiered_filters=True,
                tier_max_docs=2,
            )
            raw = execute_tool(
                ctx,
                "apply_filter",
                {
                    "filter_type": "regex",
                    "pattern": "alpha",
                    "target_set": "all",
                    "output_prefix": "alpha",
                },
            )
            payload = json.loads(raw.split("\n[Auto-sampled", 1)[0])
            self.assertEqual(payload["positive_count"], 2)
            self.assertEqual(
                payload["filter_execution"]["mode"],
                "indexed_literal_confirmed",
            )

    def test_other_regex_tools_refuse_large_active_sets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = _store(root)
            index = LexicalIndex.build(root, store)
            ctx = ToolContext(
                llm=_DummyLLM(),
                doc_store=store,
                embedding_store=None,
                embedder=None,
                use_embeddings=False,
                lexical_index=index,
                use_tiered_filters=True,
                tier_max_docs=2,
            )
            calls = [
                tool_explore_patterns(ctx, {"patterns": ["alpha"]}),
                tool_extract_field(
                    ctx,
                    {"pattern": r"Revenue: (\d+)", "save_as": "revenue"},
                ),
                tool_extract_list(
                    ctx,
                    {"pattern": r"(Alpha)", "save_as": "alpha_mentions"},
                ),
            ]
            for result in calls:
                self.assertIn("refused", result.lower())


if __name__ == "__main__":
    unittest.main()
