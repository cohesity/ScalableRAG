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

"""Size-aware filtering for Limited-Ingestion ScalableRAG."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..storage.doc_store import DocStore
from ..storage.lexical_index import LexicalIndex, LexicalIndexError
from .regex_safe import RegexTimeoutError, search_many
from .text_normalization import normalize_for_regex


class FilterBudgetExceeded(RuntimeError):
    """A corpus-wide operation would violate the exact-regex scan budget."""


def is_literal_pattern(pattern: str) -> bool:
    return bool(pattern) and not re.search(r"[.*+?\[\]()^$|\\{}]", pattern)


def literal_alternation_terms(pattern: str) -> list[str] | None:
    """Parse ``foo`` or ``foo|bar`` while rejecting actual regex syntax."""
    if not pattern:
        return None
    terms: list[str] = []
    for branch in pattern.split("|"):
        branch = branch.strip()
        if not branch or not is_literal_pattern(branch):
            return None
        terms.append(branch)
    return terms


@dataclass(frozen=True)
class TieredFilterResult:
    positive_ids: list[str]
    negative_ids: list[str] | None
    mode: str
    note: str
    target_docs: int
    target_bytes: int
    candidate_docs: int = 0
    target_was_large: bool = False
    exact: bool = True


@dataclass(frozen=True)
class TieredFilterConfig:
    max_regex_bytes: int = 10_000_000
    max_regex_docs: int = 1_000
    max_index_results: int = 16_384
    confirm_candidates: bool = True


class TieredRegexFilter:
    """Indexed literals on large sets; exact normalized regex on small sets."""

    def __init__(
        self,
        doc_store: DocStore,
        lexical_index: LexicalIndex,
        config: TieredFilterConfig | None = None,
    ):
        self.doc_store = doc_store
        self.lexical_index = lexical_index
        self.config = config or TieredFilterConfig()

    def _size(self, doc_ids: list[str]) -> tuple[bool, int, int]:
        docs = len(doc_ids)
        size = self.doc_store.set_byte_size(doc_ids)
        small = (
            docs <= self.config.max_regex_docs
            and size <= self.config.max_regex_bytes
        )
        return small, size, docs

    def _exact(self, pattern: str, doc_ids: list[str]) -> tuple[list[str], list[str]]:
        texts: list[str] = []
        present: list[str] = []
        missing: list[str] = []
        for doc_id in doc_ids:
            try:
                texts.append(normalize_for_regex(self.doc_store.get_text(doc_id)))
                present.append(doc_id)
            except KeyError:
                missing.append(doc_id)
        hits = search_many(pattern, re.IGNORECASE, texts)
        positive = [doc_id for doc_id, hit in zip(present, hits) if hit]
        negative = missing + [doc_id for doc_id, hit in zip(present, hits) if not hit]
        return positive, negative

    def apply(self, pattern: str, target_ids: list[str]) -> TieredFilterResult:
        small, target_bytes, target_docs = self._size(target_ids)
        terms = literal_alternation_terms(pattern)

        if small:
            positive, negative = self._exact(pattern, target_ids)
            return TieredFilterResult(
                positive, negative, "exact_regex",
                f"exact regex on small set ({target_docs} docs / {target_bytes / 1e6:.1f} MB)",
                target_docs, target_bytes,
            )

        if terms is None:
            raise FilterBudgetExceeded(
                f"Complex regex refused on {target_docs} documents / "
                f"{target_bytes / 1e6:.1f} MB. On a large set, first use a "
                "literal keyword or literal alternation such as 'term' or "
                "'term_a|term_b'. Apply the complex regex only after the "
                f"working set is at most {self.config.max_regex_docs} documents "
                f"and {self.config.max_regex_bytes / 1e6:.0f} MB."
            )

        try:
            candidates = self.lexical_index.search_any(
                terms,
                target_ids=target_ids,
                limit=self.config.max_index_results,
            )
        except LexicalIndexError as exc:
            raise FilterBudgetExceeded(f"Indexed literal lookup unavailable: {exc}") from exc

        if candidates.truncated:
            raise FilterBudgetExceeded(
                f"Literal query {pattern!r} matched more than "
                f"{self.config.max_index_results} documents. Add another selective "
                "literal or narrow by filename before retrying."
            )

        candidate_ids = candidates.doc_ids
        candidate_small, candidate_bytes, candidate_docs = self._size(candidate_ids)
        if self.config.confirm_candidates and candidate_small:
            positive, _ = self._exact(pattern, candidate_ids)
            return TieredFilterResult(
                positive, None, "indexed_literal_confirmed",
                f"indexed literal lookup followed by exact confirmation "
                f"({candidate_docs} candidates / {candidate_bytes / 1e6:.1f} MB)",
                target_docs, target_bytes, candidate_docs,
                target_was_large=True, exact=True,
            )

        return TieredFilterResult(
            candidate_ids, None, "indexed_literal_unconfirmed",
            "indexed literal result returned without regex confirmation because "
            f"the candidate set is still large ({candidate_docs} docs / "
            f"{candidate_bytes / 1e6:.1f} MB); narrow further before counting",
            target_docs, target_bytes, candidate_docs,
            target_was_large=True, exact=False,
        )
