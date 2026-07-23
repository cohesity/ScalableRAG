#    Copyright (c) Cohesity 2026. All rights reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""Tests for bounded user-regex execution."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scalablerag.retrieval.regex_safe import (
    RegexSafetyError,
    RegexTimeoutError,
    _run_timed,
    search_many,
    validate_user_pattern,
)


def _sleep_worker(seconds: float) -> list[bool]:
    import time

    time.sleep(seconds)
    return []
from scalablerag.retrieval.tools import ToolContext, execute_tool
from scalablerag.storage.doc_store import DocStore
from scalablerag.llm.base import BaseLLM


class _DummyLLM(BaseLLM):
    def complete(self, prompt: str, max_tokens: int = 1024, *, json_schema=None) -> str:
        return ""


class RegexSafeTests(unittest.TestCase):
    def test_rejects_nested_quantifiers(self):
        with self.assertRaises(RegexSafetyError):
            validate_user_pattern("(a+)+")

    def test_search_many_finds_simple_pattern(self):
        hits = search_many("hello", 0, ["hello world", "goodbye"])
        self.assertEqual(hits, [True, False])

    def test_run_timed_enforces_wall_clock_limit(self):
        with self.assertRaises(RegexTimeoutError):
            _run_timed(_sleep_worker, (1.0,), timeout=0.1)


class ApplyFilterRegexTests(unittest.TestCase):
    def test_apply_filter_uses_bounded_regex(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = DocStore(Path(tmp) / "docs")
            store.add_document("d1", "Alpha Corp revenue grew.", filename="a.txt")
            store.add_document("d2", "Beta Corp revenue fell.", filename="b.txt")
            ctx = ToolContext(
                llm=_DummyLLM(),
                doc_store=store,
                embedding_store=None,
                embedder=None,
                use_embeddings=False,
            )
            result = execute_tool(
                ctx,
                "apply_filter",
                {
                    "filter_type": "regex",
                    "pattern": "Alpha",
                    "target_set": "all",
                    "output_prefix": "alpha",
                },
            )
            self.assertIn("alpha_pos", result)
            self.assertIn('"positive_count": 1', result)

    def test_apply_filter_rejects_unsafe_pattern(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = DocStore(Path(tmp) / "docs")
            store.add_document("d1", "text", filename="a.txt")
            ctx = ToolContext(
                llm=_DummyLLM(),
                doc_store=store,
                embedding_store=None,
                embedder=None,
                use_embeddings=False,
            )
            result = execute_tool(
                ctx,
                "apply_filter",
                {
                    "filter_type": "regex",
                    "pattern": "(a+)+",
                    "target_set": "all",
                    "output_prefix": "bad",
                },
            )
            self.assertIn("pattern rejected", result)


if __name__ == "__main__":
    unittest.main()
