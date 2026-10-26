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

"""Tests for pattern discovery parsing and diagnostics."""

from __future__ import annotations

import json

from scalablerag.auto_patterns.discovery import PatternDiscovery
from scalablerag.storage.doc_store import DocStore


class EmptyDiscoveryLLM:
    context_length = 32_000

    def __init__(self):
        self.calls = 0

    def complete_chat(self, messages, max_tokens=1024):
        self.calls += 1
        return "[]"


def test_json_array_parser_handles_regex_character_classes():
    response = json.dumps([
        {
            "name": "section",
            "regex_template": r"(?P<body>[\s\S]{0,200})",
        }
    ])
    parsed, error = PatternDiscovery._parse_json_array_detailed(response)
    assert error is None
    assert parsed[0]["name"] == "section"


def test_sampling_does_not_alias_periodic_corpus(tmp_path):
    store = DocStore(tmp_path / "docs")
    store.add_documents(
        (
            f"hotel_{index:06d}",
            f"source hotel {index % 5}",
            f"hotel_{index:06d}.txt",
            {"source_rank": index % 5},
        )
        for index in range(100)
    )
    discovery = PatternDiscovery(EmptyDiscoveryLLM(), store, sample_size=20)
    sampled = discovery._diverse_sample(20, seed=0)
    ranks = {store.get_meta(doc_id).extra["source_rank"] for doc_id in sampled}
    assert len(ranks) > 1


def test_markdown_sections_survive_empty_llm_discovery(tmp_path):
    store = DocStore(tmp_path / "docs")
    store.add_documents(
        (
            f"doc_{index}",
            (
                f"# Hotel {index}\n\n"
                "## Facilities & Amenities\n"
                f"Pool {index}\nWi-Fi\n\n"
                "## Pricing\n"
                f"${100 + index} per night\n"
            ),
            f"doc_{index}.txt",
            None,
        )
        for index in range(20)
    )
    llm = EmptyDiscoveryLLM()
    library = PatternDiscovery(llm, store, sample_size=20).discover(verbose=False)
    names = {pattern.name for pattern in library.all_templates()}
    assert "section_facilities_amenities" in names
    assert "section_pricing" in names
    # Two rounds, each retried once after an empty response.
    assert llm.calls == 4
