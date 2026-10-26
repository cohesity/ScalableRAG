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

"""Retrieval tool implementations.

Each tool is a function that the agent can invoke. Tools operate on
the shared corpus state: doc store, embedding store, named sets,
classifiers, and the LLM.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from typing import Any

import numpy as np

from ..llm.base import BaseLLM
from ..storage.doc_store import DocStore
from ..storage.embedding_store import EmbeddingStore
from ..storage.lexical_index import LexicalIndex
from .doc_sets import DocSetRegistry
from .classifier import ClassifierStore
from .alternatives import generate_alternatives
from .regex_safe import (
    RegexSafetyError,
    RegexTimeoutError,
    compile_user_pattern,
    findall_groups_many,
    finditer_spans,
    format_safety_error,
    search_group_many,
    search_many,
    search_multi,
    validate_user_pattern,
)
from .text_normalization import normalize_for_regex as _normalize_for_regex
from .tiered_filter import (
    FilterBudgetExceeded,
    TieredFilterConfig,
    TieredRegexFilter,
    literal_alternation_terms,
)



def _unicode_tokens(text: str, min_len: int = 2) -> list[str]:
    """Tokenize text into lowercase word tokens (Unicode-aware)."""
    return [w for w in re.findall(r"\w+", text.lower(), re.UNICODE) if len(w) >= min_len]


def _match_context(text: str, pattern: "re.Pattern | None",
                   budget: int) -> str:
    """Extract the most informative snippet from *text*.

    If *pattern* is a compiled regex, find the first match and return
    surrounding context (centered on the match).  Otherwise fall back
    to the document header.
    """
    if pattern is not None:
        normalized = _normalize_for_regex(text)
        m = pattern.search(normalized)
        if m:
            # Center a window of *budget* chars around the match
            match_start = m.start()
            match_end = m.end()
            ctx_half = (budget - (match_end - match_start)) // 2
            win_start = max(0, match_start - ctx_half)
            win_end = min(len(normalized), match_end + ctx_half)
            # Expand to budget if one side was short
            if win_start == 0:
                win_end = min(len(normalized), budget)
            if win_end == len(normalized):
                win_start = max(0, len(normalized) - budget)
            snippet = normalized[win_start:win_end].strip()
            prefix = "…" if win_start > 0 else ""
            suffix = "…" if win_end < len(normalized) else ""
            return f"{prefix}{snippet}{suffix}"
    # Fallback: document header
    preview = text[:budget].rstrip()
    if len(text) > budget:
        preview += "…"
    return preview


class ToolContext:
    """Shared state available to all tools during a retrieval session."""

    def __init__(
        self,
        llm: BaseLLM,
        doc_store: DocStore,
        embedding_store: EmbeddingStore | None,
        embedder: Any,
        max_tool_result_chars: int = 200_000,
        context_length: int = 128_000,
        use_embeddings: bool = True,
        disabled_tools: frozenset[str] | None = None,
        extraction_index: Any = None,
        use_agg_tools: bool = False,
        lexical_index: LexicalIndex | None = None,
        use_tiered_filters: bool = False,
        tier_max_bytes: int = 10_000_000,
        tier_max_docs: int = 1_000,
        tier_max_results: int = 16_384,
    ):
        self.llm = llm
        self.doc_store = doc_store
        self.embedding_store = embedding_store
        self.embedder = embedder
        self.max_tool_result_chars = max_tool_result_chars
        self.context_length = context_length
        self.use_embeddings = use_embeddings
        self.disabled_tools = disabled_tools or frozenset()
        self.extraction_index = extraction_index
        self.use_agg_tools = use_agg_tools
        self.lexical_index = lexical_index
        self.use_tiered_filters = use_tiered_filters
        self.tier_filter_config = TieredFilterConfig(
            max_regex_bytes=tier_max_bytes,
            max_regex_docs=tier_max_docs,
            max_index_results=tier_max_results,
        )
        self.user_question: str = ""
        self.question_entities: list[str] = []
        self.keyword_doc_sets: dict[str, set[str]] = {}

        self._target_chars = int(context_length * 3 * 0.60)

        # Compact mode: scale sample counts for smaller context models.
        # Models with <50K context get fewer samples in tool outputs.
        self.compact = context_length < 50_000
        self.sample_count = 3 if self.compact else 6
        self.filename_max_shown = 8 if self.compact else 15
        self.count_set_max_shown = 10 if self.compact else 20

        all_ids = doc_store.all_doc_ids()
        self.sets = DocSetRegistry(all_ids)
        self.classifiers = ClassifierStore()
        self.value_sets: dict[str, dict[str, float | str]] = {}
        self._list_value_sets: set[str] = set()
        self._regex_store: dict[str, str] = {}
        self._regex_counter = 0
        self._label_cache: dict[str, dict[str, Any]] = {}
        self._read_history: set[str] = set()

    def json_dumps(self, obj: Any) -> str:
        """JSON-serialize with indentation appropriate for context budget."""
        if self.compact:
            return json.dumps(obj, separators=(",", ":"))
        return json.dumps(obj, indent=2)

    def _truncate(self, text: str) -> str:
        if len(text) > self.max_tool_result_chars:
            return (
                text[: self.max_tool_result_chars]
                + f"\n[... truncated at {self.max_tool_result_chars} chars]"
            )
        return text

    def per_doc_budget(self, n_docs: int) -> int:
        """Dynamic per-doc display budget proportional to context capacity.

        The total budget for a read_docs call is a fraction of what the
        context window can hold.  Split evenly among the docs requested.
        This naturally adapts: GPT-4.1 (1M) gets ~90K per doc for a
        3-doc read; Llama-32K gets ~3.3K.  No hard-coded thresholds.
        """
        call_budget = min(self._target_chars // 6, self.max_tool_result_chars)
        return max(2000, call_budget // max(1, n_docs))

    def regex_scan_refusal(
        self, doc_ids: list[str], operation: str
    ) -> str | None:
        """Explain why an exact runtime regex scan is disallowed, if it is."""
        if not self.use_tiered_filters:
            return None
        byte_count = self.doc_store.set_byte_size(doc_ids)
        if (
            len(doc_ids) <= self.tier_filter_config.max_regex_docs
            and byte_count <= self.tier_filter_config.max_regex_bytes
        ):
            return None
        return (
            f"{operation} refused on {len(doc_ids)} documents / "
            f"{byte_count / 1e6:.1f} MB. Limited-Ingestion permits full regex "
            "only after the active set is small. Narrow it first with an "
            "indexed literal apply_filter query, filename metadata, embeddings, "
            "or a precomputed pattern."
        )

    # ------------------------------------------------------------------
    # Auto-attach short content snippets from sets (no LLM calls).
    # These cost zero LLM calls — just memory reads on the doc store.
    # ------------------------------------------------------------------

    def snippet_budget(self) -> int:
        """Max chars for auto-enrichment snippets per tool call."""
        return 1200 if self.compact else 2400

    def entity_pattern(self) -> "re.Pattern | None":
        """Compiled regex for the question's most selective entity keyword.

        Used to center snippets on the relevant part of a document rather
        than the header.  Returns None if no entity keywords are available.
        """
        if not self.question_entities:
            return None
        try:
            return re.compile(re.escape(self.question_entities[0]), re.IGNORECASE)
        except Exception:
            return None

    def auto_snippets(self, doc_ids: list[str], n: int = 3,
                      chars_per: int = 0, label: str = "",
                      pattern: "re.Pattern | None" = None) -> str:
        """Return short content excerpts from sampled docs.

        Picks *n* docs (head + tail).  If *pattern* is given (compiled
        regex), shows context around the first match instead of the
        document header — much more informative after a regex filter.
        """
        if not doc_ids:
            return ""
        if chars_per <= 0:
            chars_per = self.snippet_budget() // max(n, 1)
        picks: list[int] = []
        if len(doc_ids) <= n:
            picks = list(range(len(doc_ids)))
        else:
            picks = [0, len(doc_ids) - 1]
            if n > 2 and len(doc_ids) > 2:
                picks.insert(1, len(doc_ids) // 2)
            picks = picks[:n]

        parts: list[str] = []
        tag = f" ({label})" if label else ""
        parts.append(f"[Auto-sampled snippets{tag} — {n} of {len(doc_ids)} docs, "
                      "explore further if needed]")
        for idx in picks:
            did = doc_ids[idx]
            try:
                meta = self.doc_store.get_meta(did)
                text = self.doc_store.get_text(did)
                preview = _match_context(text, pattern, chars_per)
                parts.append(f"  [{meta.filename}] {preview}")
            except KeyError:
                pass
        return "\n".join(parts)


# -----------------------------------------------------------------------
# Tool: read_docs
# -----------------------------------------------------------------------

def _chunk_text(text: str, chunk_size: int, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks by character count.

    Mirrors the embedding-time chunking strategy so that an embedding segment
    index can be used to read contiguous evidence at query time.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            for sep in ["\n\n", "\n", ". ", " "]:
                look_back = text[max(start, end - 200) : end]
                idx = look_back.rfind(sep)
                if idx != -1:
                    end = max(start, end - 200) + idx + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap if end < len(text) else len(text)

    return chunks


def _regex_excerpts(text: str, pattern: str, budget: int) -> str:
    """Extract windows around regex matches in a doc.

    Shows the document header plus context windows around each match
    location.  Header shrinks proportionally for small budgets to
    maximize space for match content.
    """
    header_limit = min(800, budget // 4)
    header_len = min(header_limit, len(text))
    parts = [text[:header_len]]
    budget_remaining = budget - header_len
    # Scale window size from per-doc budget.
    window = max(600, min(40_000, budget // 2))

    try:
        span_list = finditer_spans(
            pattern, re.IGNORECASE | re.DOTALL, text, max_matches=0
        )
    except RegexSafetyError:
        span_list = []
    except RegexTimeoutError:
        parts.append("[regex excerpt search timed out]")
        span_list = []
    except re.error:
        span_list = []

    seen_ranges: list[tuple[int, int]] = []
    for m_start, m_end in span_list:
        start = max(header_len, m_start - window // 2)
        end = min(len(text), m_end + window // 2)
        overlap = any(not (end < s or start > e) for s, e in seen_ranges)
        if overlap:
            continue
        excerpt = text[start:end]
        cost = len(excerpt) + 30
        if budget_remaining - cost < 0:
            break
        parts.append(f"[...match at char {m_start}...]\n{excerpt}")
        budget_remaining -= cost
        seen_ranges.append((start, end))

    if not span_list:
        leftover = text[header_len : header_len + budget_remaining]
        if leftover:
            parts.append(f"[...continued...]\n{leftover}")

    return "\n---\n".join(parts)


def _keyword_excerpts(text: str, query_tokens: set[str], budget: int) -> str:
    """Extract keyword-centred excerpts from a large doc.

    Windows are anchored on actual keyword positions (not a fixed grid)
    with asymmetric context: 1/3 before the keyword, 2/3 after.
    Answers typically follow the query term, so more post-context helps.
    """
    header_limit = min(800, budget // 4)
    header_len = min(header_limit, len(text))
    parts = [text[:header_len]]
    budget_remaining = budget - header_len

    # Contiguous reading matters for structured regions (tables, aligned text).
    # Scale window size with available budget instead of capping to a tiny size.
    window = max(2000, min(60_000, budget // 2))
    text_lower = text.lower()

    def _expand_to_line_bounds(start: int, end: int, *, extra_before_lines: int = 2) -> tuple[int, int]:
        """Expand to newline boundaries and add a few full lines of pre-context.

        Generic formatting preservation: prevents cutting structured layouts
        mid-line, and keeps nearby context immediately above a hit.
        """
        s = max(header_len, start)
        e = min(len(text), end)

        # Snap to line boundaries
        if s > header_len:
            nl = text.rfind("\n", header_len, s)
            if nl >= header_len:
                s = nl + 1
        nl2 = text.find("\n", e)
        if nl2 != -1:
            e = nl2

        # Pre-context lines, bounded by budget.
        for _ in range(extra_before_lines):
            if s <= header_len:
                break
            nl = text.rfind("\n", header_len, s - 2)
            if nl < header_len:
                s = header_len
                break
            s = nl + 1
        return s, e

    # Find all keyword positions (start index, token length, token text)
    hits: list[tuple[int, int, str]] = []
    for tok in query_tokens:
        pos = header_len
        while pos < len(text_lower):
            idx = text_lower.find(tok, pos)
            if idx < 0:
                break
            hits.append((idx, len(tok), tok))
            pos = idx + len(tok)

    if not hits:
        # Fall back to showing the document start beyond the header so the agent
        # can still make progress even when query terms don't appear verbatim.
        leftover = text[header_len : header_len + budget_remaining]
        if leftover:
            parts.append(f"[...continued...]\n{leftover}")
        return "\n---\n".join(parts)

    # Cluster nearby hits; score clusters by token coverage.
    hits.sort(key=lambda x: x[0])
    clusters: list[dict[str, Any]] = []
    current = [hits[0]]
    gap = window // 2
    # Prevent a dense token set from collapsing the whole document into one cluster.
    # This is purely geometric (char spans), not domain-specific.
    max_cluster_span = window * 3
    for h in hits[1:]:
        if (h[0] - current[-1][0] <= gap) and (h[0] - current[0][0] <= max_cluster_span):
            current.append(h)
        else:
            clusters.append({"hits": current})
            current = [h]
    clusters.append({"hits": current})

    candidates: list[tuple[float, int, int]] = []  # (score, start, end)
    for cl in clusters:
        chits: list[tuple[int, int, str]] = cl["hits"]
        cluster_start = chits[0][0]
        cluster_end = max(i + ln for i, ln, _t in chits)
        center = (cluster_start + cluster_end) // 2

        # Asymmetric window around the cluster center (keeps more post-context).
        start = max(header_len, center - window // 3)
        end = min(len(text), start + window)
        start = max(header_len, end - window)
        start, end = _expand_to_line_bounds(start, end, extra_before_lines=2)

        snippet = text[start:end]
        snippet_lower = snippet.lower()
        # Rank snippets by repeated token count (capped).
        # one-off mentions.
        token_score = sum(min(3, snippet_lower.count(t)) for t in query_tokens)
        score = float(token_score)
        candidates.append((score, start, end))

    candidates.sort(key=lambda x: -x[0])

    seen_ranges: list[tuple[int, int]] = []
    for _score, start, end in candidates[:12]:
        merged = False
        for ri, (s, e) in enumerate(seen_ranges):
            if not (end < s or start > e):
                new_s = min(s, start)
                new_e = max(e, end)
                extra = (new_e - new_s) - (e - s)
                if extra <= budget_remaining:
                    idx = next(
                        i for i, p in enumerate(parts)
                        if p.startswith(f"[...excerpt at char {s}...")
                    )
                    parts[idx] = f"[...excerpt at char {new_s}...]\n{text[new_s:new_e]}"
                    budget_remaining -= extra
                    seen_ranges[ri] = (new_s, new_e)
                merged = True
                break
        if merged:
            continue

        excerpt = text[start:end]
        if budget_remaining - len(excerpt) - 20 < 0:
            break
        parts.append(f"[...excerpt at char {start}...]\n{excerpt}")
        budget_remaining -= len(excerpt) + 20
        seen_ranges.append((start, end))

    return "\n---\n".join(parts)


def tool_read_docs(ctx: ToolContext, args: dict) -> str:
    """Read one or more documents by ID or set name.

    Supports ``offset`` to read from different parts of a set:
      offset >= 0 — start from that position (0 = beginning)
      offset < 0  — start from the end (e.g., -3 = last 3 docs)

    Set full_text=true to retrieve the complete document text without
    excerpting. Useful when you need to inspect the full structure of a
    document or when excerpts are missing the section you need.

    Large documents are shown with keyword-matched excerpts when a query
    is provided, or truncated otherwise.
    """
    raw_ids = args.get("doc_ids", [])
    set_name = args.get("set_name")
    max_docs = int(args.get("max_docs", 10))
    explicit_query = args.get("query", "")
    query = explicit_query
    offset = int(args.get("offset", 0))
    full_text = args.get("full_text", False)
    chunk_idx = args.get("chunk_idx", None)
    start_char = args.get("start_char", None)
    max_chars = args.get("max_chars", None)

    if chunk_idx is not None:
        try:
            chunk_idx = int(chunk_idx)
        except Exception:
            return "Error: chunk_idx must be an integer."

    if start_char is not None:
        try:
            start_char = int(start_char)
        except Exception:
            return "Error: start_char must be an integer."

    if max_chars is not None:
        try:
            max_chars = int(max_chars)
        except Exception:
            return "Error: max_chars must be an integer."
        if max_chars <= 0:
            return "Error: max_chars must be > 0."

    if set_name and not raw_ids:
        doc_ids = ctx.sets.resolve_ids(set_name)
        if not doc_ids:
            return (
                f"Set '{set_name}' is empty (0 docs). This usually means a "
                f"prior filter was too strict. Try a broader pattern, or use "
                f"explore_patterns to compare alternatives."
            )
    else:
        doc_ids = []
        for raw in raw_ids:
            resolved = ctx.doc_store.resolve_doc_id(raw)
            doc_ids.append(resolved if resolved else raw)

    if not doc_ids:
        return "Error: provide doc_ids or set_name."

    total_in_set = len(doc_ids)

    # Auto-populate query from user question when agent omits it
    if not query and ctx.user_question:
        query = ctx.user_question

    # Detect repeated reads with identical parameters (including query)
    read_key = (
        f"{tuple(doc_ids[:max_docs])}|{offset}|{max_docs}|{query}"
        f"|full={int(bool(full_text))}|chunk={chunk_idx}"
        f"|start={start_char}|max={max_chars}"
    )
    if read_key in ctx._read_history:
        return (
            "Already read these docs with these parameters. "
            "To see DIFFERENT content: "
            '(1) add query="specific terms" to focus excerpts on what you need, '
            "(2) use offset=-3 to read from the end, or "
            "(3) use start_char/max_chars to page through a large doc, or "
            "(4) apply_filter with a regex targeting the section you need."
        )
    ctx._read_history.add(read_key)

    if offset < 0:
        start = max(0, total_in_set + offset)
    else:
        start = min(offset, total_in_set)
    doc_ids = doc_ids[start:]

    reranked = False
    # Query-aware doc ordering (0-ingest, lexical): when the agent provides an
    # explicit query and the set has more docs than we can show, rank docs by
    # simple token overlap so the most relevant docs are surfaced first.
    #
    # Activates when: explicit query provided, offset==0, and reading excerpts
    # (not a start_char page).
    if (
        explicit_query
        and offset == 0
        and start_char is None
        and chunk_idx is None
        and not full_text
        and len(doc_ids) > max_docs
    ):
        MAX_RERANK_DOCS = 50
        if len(doc_ids) <= MAX_RERANK_DOCS:
            toks = list(dict.fromkeys(_unicode_tokens(explicit_query, min_len=3)))
            if toks:
                scored: list[tuple[float, int, str]] = []
                for idx, did in enumerate(doc_ids):
                    try:
                        t = ctx.doc_store.get_text(did).lower()
                    except KeyError:
                        continue
                    # Cap per-token contribution to avoid TOC-style repetition
                    # dominating.
                    s = float(sum(min(3, t.count(tok)) for tok in toks))
                    scored.append((s, idx, did))
                scored.sort(key=lambda x: (-x[0], x[1]))
                if scored and scored[0][0] > 0:
                    doc_ids = [did for _s, _idx, did in scored]
                    reranked = True

    if len(doc_ids) > max_docs:
        if reranked:
            note = (
                f"Set has {total_in_set} docs; reranked by explicit query relevance; "
                f"showing top {max_docs}. "
                f"Use offset/max_docs (with the same query) to page the original set order, "
                f"or narrow the set."
            )
        else:
            note = (
                f"Set has {total_in_set} docs, showing {max_docs} "
                f"starting at position {start}. "
                f"Use offset/max_docs to navigate, or narrow the set."
            )
        doc_ids = doc_ids[:max_docs]
    else:
        note = f"Showing {len(doc_ids)} docs from position {start} of {total_in_set}." if offset != 0 else ""

    budget = ctx.per_doc_budget(len(doc_ids))
    query_tokens = set(_unicode_tokens(query, min_len=3)) if query else set()
    explicit_tokens = set(_unicode_tokens(explicit_query, min_len=3)) if explicit_query else set()
    call_limit = min(ctx._target_chars // 6, ctx.max_tool_result_chars)
    total_budget = min(budget * len(doc_ids), call_limit)

    if full_text:
        budget = ctx.max_tool_result_chars
        total_budget = ctx.max_tool_result_chars

    provenance_regex = ""
    if set_name:
        provenance_regex = ctx.sets.get_filter_regex(set_name)

    chunk_chars = budget if chunk_idx is not None else None

    parts: list[str] = []
    total_chars = 0

    for did in doc_ids:
        try:
            text = ctx.doc_store.get_text(did)
            meta = ctx.doc_store.get_meta(did)
            header = f"=== {meta.filename} (id={did}, ~{meta.estimated_tokens} tokens) ==="

            if start_char is not None:
                # Contiguous slice for paging (start_char / max_chars).
                start_pos = start_char
                if start_pos < 0:
                    start_pos = max(0, len(text) + start_pos)
                start_pos = max(0, min(len(text), start_pos))
                anchor_pos = start_pos
                span = max_chars if max_chars is not None else budget
                # Snap slice bounds to line boundaries for readability, and
                # include a few full lines of pre-context so table headers are
                # visible when paging to a matched row.
                if start_pos > 0:
                    nl = text.rfind("\n", 0, start_pos)
                    if nl != -1:
                        start_pos = nl + 1
                min_start = max(0, anchor_pos - span)
                extra_before_lines = 5
                for _ in range(extra_before_lines):
                    if start_pos <= min_start:
                        start_pos = min_start
                        break
                    nl = text.rfind("\n", 0, start_pos - 2)
                    if nl == -1:
                        start_pos = min_start
                        break
                    cand = nl + 1
                    if cand < min_start:
                        start_pos = min_start
                        break
                    start_pos = cand

                end_pos = min(len(text), start_pos + span)
                nl2 = text.find("\n", end_pos)
                if nl2 != -1:
                    end_pos = nl2
                slice_text = text[start_pos:end_pos]
                nav = (
                    f"[Slice navigation: doc_chars={len(text)}; "
                    f"showing {start_pos}:{end_pos}. "
                    f"Next: start_char={end_pos} (or go back with start_char={max(0, start_pos - span)}).]"
                )
                entry = (
                    f"{header} [{len(text)} chars, showing slice chars "
                    f"{start_pos}:{end_pos}]\n{nav}\n{slice_text}\n"
                )
            elif chunk_idx is not None and chunk_chars is not None:
                chunks = _chunk_text(text, chunk_chars, overlap=200)

                if chunk_idx < 0 or chunk_idx >= len(chunks):
                    return (
                        f"Error: chunk_idx={chunk_idx} out of range for doc "
                        f"{meta.filename} ({len(chunks)} chunks)."
                    )

                chunk_text = chunks[chunk_idx]
                nav = (
                    f"[Chunk navigation: {chunk_idx + 1}/{len(chunks)}. "
                    f"Try chunk_idx={min(len(chunks) - 1, chunk_idx + 1)} "
                    f"or chunk_idx={max(0, chunk_idx - 1)}.]"
                )
                entry = (
                    f"{header} [{len(text)} chars, showing chunk "
                    f"{chunk_idx + 1}/{len(chunks)}]\n{nav}\n{chunk_text}\n"
                )
            elif len(text) <= budget:
                entry = f"{header}\n{text}\n"
            elif explicit_tokens:
                # Merge short agent query with question tokens (min length 4).
                question_tokens = (
                    set(_unicode_tokens(ctx.user_question, min_len=4))
                    if ctx.user_question
                    else set()
                )
                tok = explicit_tokens | question_tokens
                excerpts = _keyword_excerpts(text, tok, budget)
                entry = (
                    f"{header} [{len(text)} chars, showing query excerpts]\n"
                    f"{excerpts}\n"
                )
            elif provenance_regex:
                excerpts = _regex_excerpts(text, provenance_regex, budget)
                entry = (
                    f"{header} [{len(text)} chars, showing match excerpts]\n"
                    f"{excerpts}\n"
                )
            elif query_tokens:
                excerpts = _keyword_excerpts(text, query_tokens, budget)
                entry = (
                    f"{header} [{len(text)} chars, showing keyword excerpts]\n"
                    f"{excerpts}\n"
                )
            else:
                qtok = set(_unicode_tokens(ctx.user_question, min_len=3)) if ctx.user_question else set()
                excerpts = (
                    _keyword_excerpts(text, qtok, budget)
                    if qtok
                    else text[:budget]
                )
                entry = (
                    f"{header} [{len(text)} chars, showing excerpts]\n"
                    f"{excerpts}\n"
                )

            remaining_budget = total_budget - total_chars
            if len(entry) > remaining_budget:
                # Prefer returning a truncated partial doc over returning 0 docs.
                if remaining_budget <= 0:
                    shown = len(parts)
                    remaining = len(doc_ids) - shown
                    parts.append(
                        f"\n[Budget: showed {shown} of {shown + remaining} docs. "
                        f"Use max_docs={min(3, max(1, shown))} for more detail per doc, "
                        f"or offset={shown} to see the rest.]"
                    )
                    break

                trunc_note = "\n[... truncated to fit budget ...]"
                if remaining_budget > len(trunc_note) + 1:
                    entry = entry[: remaining_budget - len(trunc_note)] + trunc_note
                else:
                    entry = entry[:remaining_budget]
                parts.append(entry)
                total_chars += len(entry)
                break

            parts.append(entry)
            total_chars += len(entry)
        except KeyError:
            parts.append(f"[doc {did}: not found]")

    result = "\n".join(parts)
    if note:
        result = note + "\n\n" + result
    return ctx._truncate(result)


# -----------------------------------------------------------------------
# Tool: find_in_doc
# -----------------------------------------------------------------------

def tool_find_in_doc(ctx: ToolContext, args: dict) -> str:
    """Find occurrences of a string/regex inside a document (with char offsets).

    This is a deterministic navigation helper for large documents: use it to
    locate section headers or line items, then page contiguously with
    read_docs(start_char=..., max_chars=...).
    """
    raw_doc_id = args.get("doc_id", "")
    raw_ids = args.get("doc_ids", [])
    set_name = args.get("set_name", "")
    pattern = args.get("pattern", "")
    query = args.get("query", "")
    use_regex = bool(args.get("use_regex", False))
    case_sensitive = bool(args.get("case_sensitive", False))
    strategy = (args.get("strategy", "first") or "first").strip().lower()
    max_matches = int(args.get("max_matches", 20))
    context_chars = int(args.get("context_chars", 140))
    start_char = int(args.get("start_char", 0))
    end_char = args.get("end_char", None)
    window_chars = int(args.get("window_chars", 8000))
    overlap_chars_raw = args.get("overlap_chars", None)
    overlap_chars = int(overlap_chars_raw) if overlap_chars_raw is not None else max(0, window_chars // 2)

    if raw_doc_id and not raw_ids:
        raw_ids = [raw_doc_id]

    if set_name and not raw_ids:
        ids = ctx.sets.resolve_ids(set_name)
        if not ids:
            return f"Error: set '{set_name}' is empty or does not exist."
        raw_ids = [ids[0]]

    if not raw_ids or len(raw_ids) != 1:
        return "Error: provide exactly one doc_id (or doc_ids with one element), or set_name."
    if not pattern and not query:
        return "Error: provide pattern (substring/regex match) or query (lexical passage ranking)."
    if max_matches <= 0:
        return "Error: max_matches must be > 0."
    if context_chars < 0:
        return "Error: context_chars must be >= 0."
    if window_chars <= 0:
        return "Error: window_chars must be > 0."
    if overlap_chars < 0 or overlap_chars >= window_chars:
        return "Error: overlap_chars must be >= 0 and < window_chars."

    raw = raw_ids[0]
    resolved = ctx.doc_store.resolve_doc_id(raw) or raw

    try:
        text = ctx.doc_store.get_text(resolved)
        meta = ctx.doc_store.get_meta(resolved)
    except KeyError:
        return f"Error: doc not found: {raw!r}"

    n = len(text)
    s = max(0, min(n, start_char))
    if end_char is None:
        e = n
    else:
        try:
            e = int(end_char)
        except Exception:
            return "Error: end_char must be an integer."
        e = max(s, min(n, e))

    segment = text[s:e]

    if query and not pattern:
        # Lexical sliding windows (token overlap, no embeddings) to locate a section
        # before paging with read_docs(start_char=..., max_chars=...).
        q_tokens = list(dict.fromkeys(_unicode_tokens(query, min_len=2)))
        if not q_tokens:
            return "Error: query has no usable tokens."

        step_chars = max(1, window_chars - overlap_chars)
        windows: list[dict[str, Any]] = []
        df: dict[str, int] = {t: 0 for t in q_tokens}

        seg_n = len(segment)
        if seg_n == 0:
            return "Error: empty search range (start_char/end_char)."
        pos = 0
        while pos < seg_n:
            w0 = pos
            w1 = min(seg_n, pos + window_chars)
            wtxt = segment[w0:w1]
            toks = _unicode_tokens(wtxt, min_len=2)
            dl = len(toks)
            tf: dict[str, int] = {}
            seen: set[str] = set()
            for t in toks:
                if t in df:
                    tf[t] = tf.get(t, 0) + 1
                    seen.add(t)
            for t in seen:
                df[t] += 1
            windows.append({"seg_start": w0, "seg_end": w1, "dl": dl, "tf": tf})
            pos += step_chars

        N = len(windows)
        if N == 0:
            return "Error: no windows to score (check window_chars and search range)."
        idf: dict[str, float] = {t: math.log((N + 1) / (df_t + 1)) for t, df_t in df.items()}
        for w in windows:
            tf = w["tf"]
            score = 0.0
            for t, c in tf.items():
                score += float(c) * idf.get(t, 0.0)
            w["score"] = score

        windows.sort(key=lambda w: float(w.get("score", 0.0)), reverse=True)
        shown = [w for w in windows[:max_matches] if float(w.get("score", 0.0)) > 0.0]

        if not shown:
            wrapped = {
                "doc_id": resolved,
                "filename": meta.filename,
                "mode": "rank",
                "query": query,
                "search_range": {"start_char": s, "end_char": e},
                "window_chars": window_chars,
                "overlap_chars": overlap_chars,
                "total_windows": N,
                "matches_shown": 0,
                "matches": [],
                "hint": "No lexical matches found for query tokens in range. Try a shorter query or use pattern/use_regex.",
            }
            return ctx.json_dumps(wrapped)

        out_matches: list[dict[str, Any]] = []
        for w in shown:
            abs_s = s + int(w["seg_start"])
            abs_e = s + int(w["seg_end"])
            snippet = text[abs_s:abs_e]
            snippet_lower = snippet.lower()

            # Choose a preview centered on the densest cluster of token hits
            # (IDF-weighted) so the agent can pick the right window.
            hits: list[tuple[int, float, str]] = []
            for t in q_tokens:
                wgt = float(idf.get(t, 0.0))
                pos = 0
                step = max(1, len(t))
                while True:
                    idx = snippet_lower.find(t, pos)
                    if idx < 0:
                        break
                    hits.append((idx, wgt, t))
                    pos = idx + step

            if not hits:
                p0 = 0
                p1 = min(len(snippet), max(1, 2 * context_chars))
                hit_tokens: list[str] = []
            else:
                hits.sort(key=lambda x: x[0])
                left = 0
                total_w = 0.0
                best_w = -1.0
                best_left = 0
                best_right = 0
                for right, (pos, wgt, _tok) in enumerate(hits):
                    total_w += wgt
                    while hits[right][0] - hits[left][0] > 2 * context_chars and left < right:
                        total_w -= hits[left][1]
                        left += 1
                    if total_w > best_w:
                        best_w = total_w
                        best_left = left
                        best_right = right

                center = (hits[best_left][0] + hits[best_right][0]) // 2
                p0 = max(0, center - context_chars)
                p1 = min(len(snippet), center + context_chars)
                # Snap to line boundaries for readability (tables, aligned text).
                if p0 > 0:
                    nl = snippet.rfind("\n", 0, p0)
                    if nl != -1:
                        p0 = nl + 1
                if p1 < len(snippet):
                    nl2 = snippet.find("\n", p1)
                    if nl2 != -1:
                        p1 = nl2

                # Tokens present in the chosen preview region.
                in_preview = {t for t in q_tokens if snippet_lower.find(t, p0, p1) != -1}
                hit_tokens = sorted(in_preview)

            preview = snippet[p0:p1].strip()
            if p0 > 0:
                preview = "…" + preview
            if p1 < len(snippet):
                preview = preview + "…"

            line_num = text.count("\n", 0, abs_s) + 1
            out_matches.append(
                {
                    "char_start": abs_s,
                    "char_end": abs_e,
                    "line_num": line_num,
                    "score": round(float(w.get("score", 0.0)), 4),
                    "hit_tokens": hit_tokens,
                    "preview": preview,
                }
            )

        wrapped = {
            "doc_id": resolved,
            "filename": meta.filename,
            "mode": "rank",
            "query": query,
            "query_tokens": q_tokens,
            "search_range": {"start_char": s, "end_char": e},
            "window_chars": window_chars,
            "overlap_chars": overlap_chars,
            "total_windows": N,
            "matches_shown": len(out_matches),
            "matches": out_matches,
            "hint": "Use read_docs(start_char=<char_start>, max_chars=...) to page contiguously around a high-scoring window.",
        }
        return ctx.json_dumps(wrapped)

    # Substring/regex match mode.
    if not pattern:
        return "Error: provide pattern."

    # Collect matches as (seg_start, seg_end) in segment coordinates
    spans: list[tuple[int, int]] = []
    if use_regex:
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            span_list = finditer_spans(
                pattern, flags, segment, max_matches=0
            )
        except RegexSafetyError as exc:
            return format_safety_error(exc)
        except RegexTimeoutError as exc:
            return str(exc)
        for m_start, m_end in span_list:
            spans.append((m_start, m_end))
    else:
        hay = segment if case_sensitive else segment.lower()
        needle = pattern if case_sensitive else pattern.lower()
        if not needle:
            return "Error: empty pattern."
        pos = 0
        step = max(1, len(needle))
        while True:
            idx = hay.find(needle, pos)
            if idx < 0:
                break
            spans.append((idx, idx + len(needle)))
            pos = idx + step

    total = len(spans)
    if total == 0:
        # Helpful fallback: when an exact substring/regex match fails, provide
        # ranked windows for the same (or provided) query so the agent can still
        # navigate deterministically via start_char/max_chars.
        rank_query = (query or pattern or "").strip()
        ranked_windows: list[dict[str, Any]] = []
        if rank_query:
            q_tokens = list(dict.fromkeys(_unicode_tokens(rank_query, min_len=2)))
            if q_tokens:
                step_chars = max(1, window_chars - overlap_chars)
                windows: list[dict[str, Any]] = []
                df: dict[str, int] = {t: 0 for t in q_tokens}
                seg_n = len(segment)
                if seg_n > 0:
                    pos = 0
                    while pos < seg_n:
                        w0 = pos
                        w1 = min(seg_n, pos + window_chars)
                        wtxt = segment[w0:w1]
                        toks = _unicode_tokens(wtxt, min_len=2)
                        tf: dict[str, int] = {}
                        seen: set[str] = set()
                        for t in toks:
                            if t in df:
                                tf[t] = tf.get(t, 0) + 1
                                seen.add(t)
                        for t in seen:
                            df[t] += 1
                        windows.append({"seg_start": w0, "seg_end": w1, "tf": tf})
                        pos += step_chars
                    N = len(windows)
                    if N > 0:
                        idf: dict[str, float] = {
                            t: math.log((N + 1) / (df_t + 1)) for t, df_t in df.items()
                        }
                        for w in windows:
                            score = 0.0
                            for t, c in (w.get("tf") or {}).items():
                                score += float(c) * idf.get(t, 0.0)
                            w["score"] = score
                        windows.sort(key=lambda w: float(w.get("score", 0.0)), reverse=True)
                        for w in windows[: min(max_matches, 5)]:
                            if float(w.get("score", 0.0)) <= 0.0:
                                continue
                            abs_s = s + int(w["seg_start"])
                            abs_e = s + int(w["seg_end"])
                            snippet = text[abs_s:abs_e]
                            snippet_lower = snippet.lower()
                            hits: list[tuple[int, float, str]] = []
                            for t in q_tokens:
                                wgt = float(idf.get(t, 0.0))
                                pos = 0
                                step = max(1, len(t))
                                while True:
                                    idx = snippet_lower.find(t, pos)
                                    if idx < 0:
                                        break
                                    hits.append((idx, wgt, t))
                                    pos = idx + step

                            if not hits:
                                p0 = 0
                                p1 = min(len(snippet), max(1, 2 * context_chars))
                                hit_tokens: list[str] = []
                            else:
                                hits.sort(key=lambda x: x[0])
                                left = 0
                                total_w = 0.0
                                best_w = -1.0
                                best_left = 0
                                best_right = 0
                                for right, (pos, wgt, _tok) in enumerate(hits):
                                    total_w += wgt
                                    while hits[right][0] - hits[left][0] > 2 * context_chars and left < right:
                                        total_w -= hits[left][1]
                                        left += 1
                                    if total_w > best_w:
                                        best_w = total_w
                                        best_left = left
                                        best_right = right

                                center = (hits[best_left][0] + hits[best_right][0]) // 2
                                p0 = max(0, center - context_chars)
                                p1 = min(len(snippet), center + context_chars)
                                if p0 > 0:
                                    nl = snippet.rfind("\n", 0, p0)
                                    if nl != -1:
                                        p0 = nl + 1
                                if p1 < len(snippet):
                                    nl2 = snippet.find("\n", p1)
                                    if nl2 != -1:
                                        p1 = nl2

                                in_preview = {t for t in q_tokens if snippet_lower.find(t, p0, p1) != -1}
                                hit_tokens = sorted(in_preview)

                            preview = snippet[p0:p1].strip()
                            if p0 > 0:
                                preview = "…" + preview
                            if p1 < len(snippet):
                                preview = preview + "…"
                            line_num = text.count("\n", 0, abs_s) + 1
                            ranked_windows.append(
                                {
                                    "char_start": abs_s,
                                    "char_end": abs_e,
                                    "line_num": line_num,
                                    "score": round(float(w.get("score", 0.0)), 4),
                                    "hit_tokens": hit_tokens,
                                    "preview": preview,
                                }
                            )

        wrapped = {
            "doc_id": resolved,
            "filename": meta.filename,
            "pattern": pattern,
            "use_regex": use_regex,
            "case_sensitive": case_sensitive,
            "search_range": {"start_char": s, "end_char": e},
            "total_matches": 0,
            "matches_shown": 0,
            "matches": [],
            "ranked_windows": ranked_windows,
            "hint": (
                "No exact matches. Try a shorter substring, a different spelling, or use_regex=true. "
                + (
                    "Ranked windows are provided as a fallback—page with read_docs(start_char=..., max_chars=...)."
                    if ranked_windows else
                    "If you only have a natural-language description, try find_in_doc(query=...) to rank windows."
                )
            ),
        }
        return ctx.json_dumps(wrapped)

    # If too many matches, default to showing the earliest ones in order.
    # Optionally use a "spread" strategy to cover the range.
    if strategy not in {"first", "spread", "last"}:
        return "Error: strategy must be one of: first, spread, last."

    if total <= max_matches:
        pick = list(range(total))
    elif strategy == "first":
        pick = list(range(max_matches))
    elif strategy == "last":
        pick = list(range(total - max_matches, total))
    else:
        # spread
        if max_matches == 1:
            pick = [0]
        else:
            pick = []
            for k in range(max_matches):
                i = int(round(k * (total - 1) / (max_matches - 1)))
                if not pick or i != pick[-1]:
                    pick.append(i)

    out_matches: list[dict[str, Any]] = []
    for i in pick:
        ms, me = spans[i]
        abs_s = s + ms
        abs_e = s + me
        ctx_s = max(s, abs_s - context_chars)
        ctx_e = min(e, abs_e + context_chars)
        snippet = text[ctx_s:ctx_e]
        pfx = "…" if ctx_s > s else ""
        sfx = "…" if ctx_e < e else ""
        preview = (pfx + snippet + sfx).strip()
        # Compute 1-indexed line number.
        line_num = text.count("\n", 0, abs_s) + 1
        out_matches.append(
            {
                "char_start": abs_s,
                "char_end": abs_e,
                "line_num": line_num,
                "preview": preview,
            }
        )

    wrapped = {
        "doc_id": resolved,
        "filename": meta.filename,
        "pattern": pattern,
        "use_regex": use_regex,
        "case_sensitive": case_sensitive,
        "strategy": strategy,
        "search_range": {"start_char": s, "end_char": e},
        "total_matches": total,
        "matches_shown": len(out_matches),
        "matches": out_matches,
        "hint": (
            "Use read_docs(start_char=<char_start>, max_chars=...) to page to a match. "
            "If there are many matches, narrow with start_char/end_char or set strategy=first."
        ),
    }
    return ctx.json_dumps(wrapped)


# -----------------------------------------------------------------------
# Tool: table_lookup
# -----------------------------------------------------------------------

def tool_table_lookup(ctx: ToolContext, args: dict) -> str:
    """Locate a table-like row in a document and return nearby header context.

    This is an ingestion-free, deterministic helper for long, structured docs
    (financial statements, schedules, aligned text). It returns:
      - the matched row line(s)
      - a few lines above (often containing column headers/years)
      - parsed years and numeric tokens when possible

    Use with read_docs(start_char=..., max_chars=...) to page contiguously if
    you need more surrounding context.
    """
    raw_doc_id = args.get("doc_id", "")
    raw_ids = args.get("doc_ids", [])
    set_name = args.get("set_name", "")
    row = (args.get("row", "") or args.get("pattern", "") or "").strip()
    use_regex = bool(args.get("use_regex", False))
    case_sensitive = bool(args.get("case_sensitive", False))
    max_matches = int(args.get("max_matches", 10))
    header_lines = int(args.get("header_lines", 8))
    context_lines = int(args.get("context_lines", 2))
    year_scan_lines = int(args.get("year_scan_lines", 200))
    start_char = int(args.get("start_char", 0))
    end_char = args.get("end_char", None)

    if raw_doc_id and not raw_ids:
        raw_ids = [raw_doc_id]
    if set_name and not raw_ids:
        ids = ctx.sets.resolve_ids(set_name)
        if not ids:
            return f"Error: set '{set_name}' is empty or does not exist."
        raw_ids = [ids[0]]

    if not raw_ids or len(raw_ids) != 1:
        return "Error: provide exactly one doc_id (or doc_ids with one element), or set_name."
    if not row:
        return "Error: provide row (or pattern)."
    if max_matches <= 0:
        return "Error: max_matches must be > 0."
    if header_lines < 0:
        return "Error: header_lines must be >= 0."
    if context_lines < 0:
        return "Error: context_lines must be >= 0."
    if year_scan_lines < 0:
        return "Error: year_scan_lines must be >= 0."

    raw = raw_ids[0]
    resolved = ctx.doc_store.resolve_doc_id(raw) or raw
    try:
        text = ctx.doc_store.get_text(resolved)
        meta = ctx.doc_store.get_meta(resolved)
    except KeyError:
        return f"Error: doc not found: {raw!r}"

    n = len(text)
    s = max(0, min(n, start_char))
    if end_char is None:
        e = n
    else:
        try:
            e = int(end_char)
        except Exception:
            return "Error: end_char must be an integer."
        e = max(s, min(n, e))

    segment = text[s:e]
    if not segment:
        return "Error: empty search range (start_char/end_char)."

    # Split into lines with offsets inside segment.
    lines = segment.splitlines(keepends=True)
    line_starts: list[int] = []
    pos = 0
    for ln in lines:
        line_starts.append(pos)
        pos += len(ln)

    # Prepare matcher.
    matched_line_indices: set[int] | None = None
    if use_regex:
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            line_hits = search_many(row, flags, lines)
        except RegexSafetyError as exc:
            return format_safety_error(exc)
        except RegexTimeoutError as exc:
            return str(exc)
        matched_line_indices = {i for i, hit in enumerate(line_hits) if hit}
    else:
        needle = row if case_sensitive else row.lower()
        def _is_match(line: str) -> bool:
            hay = line if case_sensitive else line.lower()
            return needle in hay

    # Helper parsers
    year_rx = re.compile(r"\b(19|20)\d{2}\b")
    num_rx = re.compile(r"\(?-?\$?\s*\d[\d,]*\.?\d*\)?%?")

    matches: list[dict[str, Any]] = []
    for i, ln in enumerate(lines):
        if matched_line_indices is not None:
            if i not in matched_line_indices:
                continue
        elif not _is_match(ln):
            continue

        # Collect a small block: header above + row + a couple continuation lines.
        h0 = max(0, i - header_lines)
        c1 = min(len(lines), i + 1 + context_lines)
        header_block = "".join(lines[h0:i]).rstrip("\n")
        row_block = "".join(lines[i:c1]).rstrip("\n")

        # Absolute offsets (into full doc) for the start of the matched line.
        abs_line_start = s + line_starts[i]
        abs_line_end = abs_line_start + len(ln.rstrip("\n"))
        line_num = text.count("\n", 0, abs_line_start) + 1

        # Try to parse years from the header area (prefer the line with most years).
        # Use a wider scan range than header_block to reliably capture column headers.
        scan0 = max(0, i - year_scan_lines)
        header_lines_list = [lines[k].rstrip("\n") for k in range(scan0, i)]
        best_years: list[str] = []
        best_year_line: str = ""
        best_year_score: float = float("-inf")
        for hl in header_lines_list:
            yrs2_raw = [m.group(0) for m in year_rx.finditer(hl)]
            if not yrs2_raw:
                continue
            # Dedup while preserving order.
            seen_y: set[str] = set()
            yrs2: list[str] = []
            for y in yrs2_raw:
                if y not in seen_y:
                    seen_y.add(y)
                    yrs2.append(y)
            if len(yrs2) < 2:
                continue

            letters = sum(1 for c in hl if c.isalpha())
            digits = sum(1 for c in hl if c.isdigit())
            length = len(hl.strip())
            word_count = len(re.findall(r"[A-Za-z]{2,}", hl))
            # Prefer compact, table-like header lines (few letters, lots of digits).
            # Penalize very long narrative lines that mention multiple years.
            score = (len(yrs2) * 100.0) + (digits * 2.0) - (letters * 1.0)
            if length > 140:
                score -= (length - 140) * 3.0
            if letters > digits + 10:
                score -= 50.0
            if word_count > 12:
                score -= (word_count - 12) * 20.0

            if score > best_year_score:
                best_year_score = score
                best_years = yrs2
                best_year_line = hl

        # Parse numeric tokens from the matched row block.
        # Many filings put the row label on one line and the year columns on
        # the following line(s); we collect numbers until a non-numeric line
        # with letters appears.
        row_lines = [l.rstrip("\n") for l in row_block.splitlines()]
        num_tokens: list[str] = []
        num_positions: list[int] = []
        for li, row_ln in enumerate(row_lines):
            stripped = row_ln.strip()
            if li > 0 and re.search(r"[A-Za-z]", stripped):
                break
            for m in num_rx.finditer(row_ln):
                tok = m.group(0).strip()
                if not tok:
                    continue
                if tok in {"$", "(", ")"}:
                    continue
                num_tokens.append(tok)
                if li == 0:
                    num_positions.append(m.start())

        # Map years->values when counts match (simple, but often correct for statements).
        year_to_value: dict[str, str] = {}
        if best_years and len(best_years) == len(num_tokens) and len(best_years) <= 6:
            for y, v in zip(best_years, num_tokens):
                year_to_value[y] = v

        matches.append(
            {
                "char_start": abs_line_start,
                "char_end": abs_line_end,
                "line_num": line_num,
                "row_block": row_block,
                "header_block": header_block,
                "parsed": {
                    "years": best_years,
                    "year_header_line": best_year_line,
                    "year_header_score": None if best_year_score == float("-inf") else round(best_year_score, 2),
                    "row_numbers": num_tokens,
                    "row_number_positions": num_positions,
                    "year_to_value": year_to_value,
                },
            }
        )
        if len(matches) >= max_matches:
            break

    wrapped = {
        "doc_id": resolved,
        "filename": meta.filename,
        "row": row,
        "use_regex": use_regex,
        "case_sensitive": case_sensitive,
        "search_range": {"start_char": s, "end_char": e},
        "total_matches": len(matches),
        "matches": matches,
        "hint": (
            "Use read_docs(start_char=<char_start>, max_chars=...) to page contiguously around a match. "
            "If multiple matches exist, disambiguate by nearby header_block or restrict the search_range."
        ),
    }
    return ctx.json_dumps(wrapped)


# -----------------------------------------------------------------------
# Tool: label_docs
# -----------------------------------------------------------------------


def tool_label_docs(ctx: ToolContext, args: dict) -> str:
    """Read documents and label them for specified conditions (0/1).

    Returns JSON with labels per doc per condition. Results are cached
    for use with train_classifier.
    """
    doc_ids = args.get("doc_ids", [])
    set_name = args.get("set_name")
    labels = args.get("labels", [])
    max_docs = int(args.get("max_docs", 20))

    if set_name and not doc_ids:
        doc_ids = ctx.sets.resolve_ids(set_name)

    if not doc_ids:
        return "Error: provide doc_ids or set_name."
    if not labels:
        return "Error: provide at least one label description."

    if len(doc_ids) > max_docs:
        note = f"Set has {len(doc_ids)} docs, labeling first {max_docs}."
        doc_ids = doc_ids[:max_docs]
    else:
        note = ""

    label_descriptions = "\n".join(
        f"  {i+1}. {lbl}" for i, lbl in enumerate(labels)
    )

    results: dict[str, dict] = {}
    for did in doc_ids:
        try:
            text = ctx.doc_store.get_text(did)
            meta = ctx.doc_store.get_meta(did)
        except KeyError:
            results[did] = {"error": "not found"}
            continue

        max_chars = min(100_000, ctx.llm.context_length * 3)
        doc_text = text[:max_chars] if len(text) > max_chars else text

        prompt = (
            f"You are labeling a document based on conditions.\n\n"
            f"Document (filename: {meta.filename}):\n"
            f"---\n{doc_text}\n---\n\n"
            f"For each condition below, respond with 1 (true) or 0 (false):\n"
            f"{label_descriptions}\n\n"
            f"Return ONLY a JSON object mapping each condition number (as string) "
            f"to 0 or 1. Example: {{\"1\": 1, \"2\": 0}}\n"
            f"Return JSON only, no other text."
        )

        fits, _, _ = ctx.llm.prompt_fits(prompt, 256)
        if not fits:
            doc_text = text[: max_chars // 4]
            prompt = (
                f"You are labeling a document based on conditions.\n\n"
                f"Document (filename: {meta.filename}, TRUNCATED):\n"
                f"---\n{doc_text}\n---\n\n"
                f"For each condition below, respond with 1 (true) or 0 (false):\n"
                f"{label_descriptions}\n\n"
                f"Return ONLY a JSON object mapping each condition number (as string) "
                f"to 0 or 1.\nReturn JSON only."
            )

        try:
            resp = ctx.llm.complete(prompt, max_tokens=256)
            resp = resp.strip()
            if resp.startswith("```"):
                resp = resp.split("```")[1]
                if resp.startswith("json"):
                    resp = resp[4:]
            label_dict = json.loads(resp)
        except Exception as e:
            label_dict = {"error": str(e)}

        named_labels = {}
        for i, lbl in enumerate(labels):
            key = str(i + 1)
            val = label_dict.get(key, label_dict.get(i + 1, None))
            named_labels[lbl] = int(val) if val is not None else None
        results[did] = {"filename": meta.filename, "labels": named_labels}

    cache_key = f"label_{len(ctx._label_cache)}"
    ctx._label_cache[cache_key] = results

    out = {"label_batch_id": cache_key, "results": results}
    if note:
        out["note"] = note
    return json.dumps(out, indent=2)


# -----------------------------------------------------------------------
# Tool: create_regex
# -----------------------------------------------------------------------


def tool_create_regex(ctx: ToolContext, args: dict) -> str:
    """Create or register a regex to detect a condition in documents.

    Two modes:
      (a) Provide ``pattern`` directly — the agent writes its own regex.
      (b) Provide ``condition`` — the LLM generates a regex.

    Either way the pattern is stored, tested on samples, and returned.
    """
    pattern = args.get("pattern", "")
    condition = args.get("condition", "")
    sample_ids = args.get("sample_doc_ids", [])
    sample_set = args.get("sample_set_name")
    num_samples = int(args.get("num_samples", 5))

    if not pattern and not condition:
        return "Error: provide either 'pattern' (your own regex) or 'condition' (LLM creates regex)."

    # Resolve sample docs
    if sample_set and not sample_ids:
        sample_ids = ctx.sets.resolve_ids(sample_set)
    if not sample_ids:
        sample_ids = ctx.doc_store.all_doc_ids()
    sample_ids = sample_ids[:num_samples]

    if not pattern:
        # Mode (b): LLM generates the regex
        sample_texts = []
        for did in sample_ids:
            try:
                text = ctx.doc_store.get_text(did)[:8000]
                meta = ctx.doc_store.get_meta(did)
                sample_texts.append(f"--- {meta.filename} ---\n{text}\n")
            except KeyError:
                continue

        samples_str = "\n".join(sample_texts)

        prompt = (
            f"Create a Python regex pattern (re.IGNORECASE) that detects "
            f"the following condition in documents:\n\n"
            f"  Condition: {condition}\n\n"
            f"Sample documents:\n{samples_str}\n\n"
            f"IMPORTANT:\n"
            f"- Keep the regex SIMPLE. A short keyword or alternation is "
            f"better than a complex multi-part pattern.\n"
            f"- Prefer high recall (catch everything relevant) over "
            f"precision. False positives can be filtered later.\n"
            f"- Look at how the information actually appears in the "
            f"sample documents and match THAT format.\n\n"
            f"Return ONLY a JSON object:\n"
            f"  {{\"pattern\": \"...\", \"explanation\": \"...\"}}\n"
            f"Return JSON only, no other text."
        )

        try:
            resp = ctx.llm.complete(prompt, max_tokens=512)
            resp = resp.strip()
            if resp.startswith("```"):
                resp = resp.split("```")[1]
                if resp.startswith("json"):
                    resp = resp[4:]
            result = json.loads(resp)
            pattern = result.get("pattern", "")
            explanation = result.get("explanation", "")
        except Exception as e:
            return f"Error creating regex: {e}"
    else:
        explanation = f"Agent-provided pattern"

    try:
        compile_user_pattern(pattern, re.IGNORECASE)
    except RegexSafetyError as exc:
        return format_safety_error(exc)
    except re.error as e:
        return f"Invalid regex: {pattern!r} — {e}"

    ctx._regex_counter += 1
    regex_id = f"regex_{ctx._regex_counter}"
    ctx._regex_store[regex_id] = pattern

    # Test on samples
    test_results = []
    sample_texts: list[str] = []
    sample_dids: list[str] = []
    for did in sample_ids:
        try:
            sample_texts.append(_normalize_for_regex(ctx.doc_store.get_text(did)))
            sample_dids.append(did)
        except KeyError:
            continue

    try:
        sample_hits = search_many(pattern, re.IGNORECASE, sample_texts)
    except RegexTimeoutError as exc:
        return str(exc)

    for did, match in zip(sample_dids, sample_hits):
        try:
            meta = ctx.doc_store.get_meta(did)
            test_results.append(
                {"doc_id": did, "filename": meta.filename, "match": match}
            )
        except KeyError:
            continue

    matched = sum(1 for t in test_results if t["match"])

    return json.dumps(
        {
            "regex_id": regex_id,
            "pattern": pattern,
            "explanation": explanation,
            "sample_test": test_results,
            "matched_samples": f"{matched}/{len(test_results)}",
        },
        indent=2,
    )


# -----------------------------------------------------------------------
# Tool: train_classifier
# -----------------------------------------------------------------------


def tool_train_classifier(ctx: ToolContext, args: dict) -> str:
    """Train a binary classifier on document embeddings + labels."""
    classifier_type = args.get("classifier_type", "xgboost")
    classifier_id = args.get("classifier_id")

    labels_dict: dict[str, int] = {}
    if "labels" in args and args["labels"]:
        labels_dict = {str(k): int(v) for k, v in args["labels"].items()}
    elif "label_batch_id" in args:
        batch_id = args["label_batch_id"]
        label_key = args.get("label_key", "")
        if batch_id not in ctx._label_cache:
            return f"Error: label_batch_id {batch_id!r} not found."
        batch = ctx._label_cache[batch_id]
        for did, info in batch.items():
            if "labels" in info and label_key in info["labels"]:
                val = info["labels"][label_key]
                if val is not None:
                    labels_dict[did] = int(val)
    else:
        return "Error: provide labels dict or label_batch_id + label_key."

    if len(labels_dict) < 4:
        return f"Error: need at least 4 labeled docs, got {len(labels_dict)}."

    pos = sum(1 for v in labels_dict.values() if v == 1)
    neg = len(labels_dict) - pos
    if pos == 0 or neg == 0:
        return (
            f"Error: need both positive and negative examples. "
            f"Got {pos} positive, {neg} negative."
        )

    doc_ids = list(labels_dict.keys())
    y = np.array([labels_dict[d] for d in doc_ids], dtype=np.float32)

    try:
        X = ctx.embedding_store.get_embeddings_for_ids(doc_ids)
    except KeyError as e:
        return f"Error getting embeddings: {e}"

    try:
        result = ctx.classifiers.train(
            classifier_type=classifier_type,
            embeddings=X,
            labels=y,
            doc_ids=doc_ids,
            classifier_id=classifier_id,
        )
    except Exception as e:
        return f"Error training classifier: {e}"

    return json.dumps(
        {
            "classifier_id": result.classifier_id,
            "type": result.classifier_type,
            "train_accuracy": round(result.accuracy, 4),
            "train_size": result.num_train,
            "positive": result.num_positive,
            "negative": result.num_negative,
            "feature_dim": result.feature_dim,
        },
        indent=2,
    )


# -----------------------------------------------------------------------
# Tool: apply_filter
# -----------------------------------------------------------------------


def tool_apply_filter(ctx: ToolContext, args: dict) -> str:
    """Apply a classifier or regex to a document set.

    For regex, you can either reference a stored regex_id (from
    create_regex) or provide a ``pattern`` string directly.

    Creates <prefix>_pos and <prefix>_neg named sets.
    """
    filter_type = args.get("filter_type", "")
    filter_id = args.get("filter_id", "")
    inline_pattern = args.get("pattern", "")
    target_set = args.get("target_set", "all")
    output_prefix = args.get("output_prefix", filter_id or "filter")

    if not filter_type:
        return "Error: provide filter_type ('classifier' or 'regex')."

    target_ids = ctx.sets.resolve_ids(target_set)
    if not target_ids:
        return f"Error: target set {target_set!r} is empty."

    positive_ids: list[str] = []
    negative_ids: list[str] = []
    compiled_re: re.Pattern | None = None
    tiered_decision = None

    if filter_type == "classifier":
        if not filter_id or filter_id not in ctx.classifiers._classifiers:
            return f"Error: classifier {filter_id!r} not found."

        try:
            X = ctx.embedding_store.get_embeddings_for_ids(target_ids)
            preds = ctx.classifiers.predict(filter_id, X)
        except Exception as e:
            return f"Error applying classifier: {e}"

        for i, did in enumerate(target_ids):
            if preds[i] == 1:
                positive_ids.append(did)
            else:
                negative_ids.append(did)

    elif filter_type == "regex":
        # Resolve the pattern: inline > stored
        if inline_pattern:
            pattern = inline_pattern
            # Auto-register for potential reuse
            ctx._regex_counter += 1
            auto_id = f"regex_{ctx._regex_counter}"
            ctx._regex_store[auto_id] = pattern
            if not filter_id:
                filter_id = auto_id
        elif filter_id and filter_id in ctx._regex_store:
            pattern = ctx._regex_store[filter_id]
        else:
            return (
                "Error: for regex filter, provide either 'pattern' (inline) "
                "or 'filter_id' referencing a stored regex."
            )

        try:
            safe_compiled = compile_user_pattern(pattern, re.IGNORECASE)
        except RegexSafetyError as exc:
            return format_safety_error(exc)

        if ctx.use_tiered_filters:
            if ctx.lexical_index is None:
                return (
                    "Error: tiered filtering requires a lexical index. "
                    "Re-run ingestion with --lexical-index."
                )
            policy = TieredRegexFilter(
                ctx.doc_store, ctx.lexical_index, ctx.tier_filter_config
            )
            try:
                tiered_decision = policy.apply(pattern, target_ids)
            except FilterBudgetExceeded as exc:
                return f"Filter refused: {exc}"
            except RegexTimeoutError as exc:
                return str(exc)
            positive_ids.extend(tiered_decision.positive_ids)
            if tiered_decision.negative_ids is not None:
                negative_ids.extend(tiered_decision.negative_ids)
        else:
            rows_are_normalized = hasattr(ctx.doc_store, "iter_normalized_texts")

            def _text_rows():
                if rows_are_normalized:
                    yield from ctx.doc_store.iter_normalized_texts(target_ids)
                    return
                if hasattr(ctx.doc_store, "iter_texts"):
                    yield from ctx.doc_store.iter_texts(target_ids)
                    return
                for did in target_ids:
                    try:
                        yield did, ctx.doc_store.get_text(did)
                    except KeyError:
                        negative_ids.append(did)

            batch_ids: list[str] = []
            batch_texts: list[str] = []

            def _scan_batch() -> str | None:
                if not batch_ids:
                    return None
                # Literal alternations cannot catastrophically backtrack.
                # Scan them directly so the Zero baseline measures corpus
                # search rather than serializing every batch to a subprocess.
                if literal_alternation_terms(pattern) is not None:
                    hits = [safe_compiled.search(text) is not None for text in batch_texts]
                else:
                    try:
                        hits = search_many(pattern, re.IGNORECASE, batch_texts)
                    except RegexTimeoutError as exc:
                        return str(exc)
                for did, hit in zip(batch_ids, hits):
                    (positive_ids if hit else negative_ids).append(did)
                batch_ids.clear()
                batch_texts.clear()
                return None

            for did, text in _text_rows():
                batch_ids.append(did)
                batch_texts.append(
                    text if rows_are_normalized else _normalize_for_regex(text)
                )
                if len(batch_ids) >= 10_000:
                    error = _scan_batch()
                    if error:
                        return error
            error = _scan_batch()
            if error:
                return error

        compiled_re = safe_compiled
    else:
        return f"Error: filter_type must be 'classifier' or 'regex', got {filter_type!r}."

    pos_name = f"{output_prefix}_pos"
    neg_name = f"{output_prefix}_neg"
    regex_prov = pattern if filter_type == "regex" else ""
    ctx.sets.register(
        pos_name, positive_ids,
        source=f"{filter_type}:{filter_id}",
        description=f"Docs matching {filter_id}",
        filter_regex=regex_prov,
    )
    lazy_negative = bool(
        tiered_decision is not None
        and tiered_decision.negative_ids is None
    )
    if lazy_negative:
        ctx.sets.register_complement(
            neg_name, target_set, positive_ids,
            source=f"{filter_type}:{filter_id}",
            description=f"Docs not matching {filter_id}",
            filter_regex=regex_prov,
        )
    else:
        ctx.sets.register(
            neg_name, negative_ids,
            source=f"{filter_type}:{filter_id}",
            description=f"Docs not matching {filter_id}",
            filter_regex=regex_prov,
        )

    sample_n = ctx.sample_count

    def _diverse_samples(ids: list[str], n: int = sample_n) -> list[dict]:
        """Sample doc entries from beginning, middle, and end to avoid bias."""
        samples: list[dict] = []
        if not ids:
            return samples
        if len(ids) <= n:
            indices = list(range(len(ids)))
        else:
            third = n // 3
            head = list(range(third))
            mid_start = len(ids) // 2 - third // 2
            mid = list(range(mid_start, mid_start + third))
            tail = list(range(len(ids) - (n - 2 * third), len(ids)))
            indices = sorted(set(head + mid + tail))
        for i in indices:
            try:
                meta = ctx.doc_store.get_meta(ids[i])
                samples.append({"doc_id": ids[i], "filename": meta.filename})
            except (KeyError, IndexError):
                pass
        return samples

    result: dict = {
        "positive_set": pos_name,
        "positive_count": len(positive_ids),
        "negative_set": neg_name,
        "negative_count": (
            len(target_ids) - len(positive_ids)
            if lazy_negative else len(negative_ids)
        ),
        "total_filtered": len(target_ids),
        "sample_positive_docs": _diverse_samples(positive_ids),
        "sample_negative_docs": _diverse_samples(negative_ids),
    }
    if tiered_decision is not None:
        result["filter_execution"] = {
            "mode": tiered_decision.mode,
            "exact": tiered_decision.exact,
            "note": tiered_decision.note,
            "target_docs": tiered_decision.target_docs,
            "target_bytes": tiered_decision.target_bytes,
            "candidate_docs": tiered_decision.candidate_docs,
        }

    if len(positive_ids) == 0:
        hint = (
            f"Zero matches on {len(target_ids)} docs. "
            f"Try: (1) a simpler/shorter regex, "
            f"(2) search_filenames if the condition is about "
            f"document names"
        )
        if ctx.use_embeddings:
            hint += ", or (3) cosine_search for semantic matching."
        else:
            hint += "."
        result["zero_result_hint"] = hint

    # This is only an advisory diagnostic.  Never turn it into a second
    # corpus-wide scan at scale: filename search is not part of the filter's
    # semantics, and the diagnostic otherwise dominates indexed retrieval.
    if (
        filter_type == "regex"
        and compiled_re
        and len(positive_ids) > 0
        and len(target_ids) <= 100_000
    ):
        if hasattr(ctx.doc_store, "count_filename_matches"):
            fn_match_count = ctx.doc_store.count_filename_matches(
                pattern, target_ids
            )
        else:
            fn_match_count = sum(
                1 for did in target_ids
                if compiled_re.search(
                    ctx.doc_store.get_meta(did).filename
                )
            )
        if fn_match_count != len(positive_ids):
            result["scope_check"] = (
                f"apply_filter searches DOCUMENT TEXT. "
                f"{len(positive_ids)} docs matched in text, but "
                f"{fn_match_count} filenames match the same pattern. "
                f"If you intended to filter by filename/date, "
                f"use search_filenames instead."
            )

    # Alternation lint: detect ungrouped top-level | that likely
    # misses the intended grouping (e.g., "A.*B|C" → "(A.*B)|(C)").
    if filter_type == "regex" and "|" in pattern:
        depth = 0
        top_splits: list[str] = []
        buf: list[str] = []
        for ch in pattern:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            elif ch == "|" and depth == 0:
                top_splits.append("".join(buf))
                buf = []
                continue
            buf.append(ch)
        top_splits.append("".join(buf))
        if len(top_splits) >= 2:
            lens = [len(s.strip()) for s in top_splits]
            if max(lens) > 2 * min(max(1, l) for l in lens):
                result["alternation_note"] = (
                    f"Regex has top-level '|' — branches are "
                    f"independent alternatives: {top_splits}. "
                    f"Use parentheses to group alternatives: "
                    f"'prefix.*(opt1|opt2)' not "
                    f"'prefix.*opt1|opt2'."
                )

    # Keyword decomposition: for every regex filter, show how many docs
    # each individual keyword matches.  This is pure information — no
    # threshold gate — and lets the agent see the selectivity landscape
    # of its pattern at every step.
    if (
        filter_type == "regex"
        and target_ids
        and not (tiered_decision and tiered_decision.target_was_large)
    ):
        keywords = list(dict.fromkeys(
            re.findall(r'[^\W_]{4,}', pattern, re.UNICODE)))
        if len(keywords) >= 2:
            probes: list[dict] = []
            seen: set[str] = set()
            probe_kws: list[str] = []
            for kw in keywords[:4]:
                if kw.lower() in seen:
                    continue
                seen.add(kw.lower())
                probe_kws.append(kw)

            norm_texts: list[str] = []
            for did in target_ids:
                try:
                    norm_texts.append(
                        _normalize_for_regex(ctx.doc_store.get_text(did))
                    )
                except KeyError:
                    norm_texts.append("")
            try:
                kw_rows = []
                for kw in probe_kws:
                    try:
                        kw_rows.append(
                            search_many(kw, re.IGNORECASE, norm_texts)
                        )
                    except (RegexSafetyError, RegexTimeoutError):
                        kw_rows.append([])
            except Exception:
                kw_rows = []

            for kw, row in zip(probe_kws, kw_rows):
                cnt = sum(1 for hit in row if hit)
                probes.append({"keyword": kw, "matches": cnt,
                               "pct": round(100 * cnt / len(target_ids), 1)})
            if probes:
                result["keyword_decomposition"] = probes

                matching_kws = [p for p in probes if p["matches"] > 0]
                if matching_kws:
                    min_kw = min(p["matches"] for p in matching_kws)
                    max_kw = max(p["matches"] for p in matching_kws)
                    if min_kw > len(positive_ids):
                        result["selectivity_note"] = (
                            f"Combined pattern matched {len(positive_ids)}; "
                            f"individual keywords match "
                            f"{min_kw}\u2013{max_kw} each. "
                            f"Consider starting with one keyword."
                        )
                if len(matching_kws) >= 2:
                    kw1, kw2 = matching_kws[0]["keyword"], matching_kws[1]["keyword"]
                    try:
                        both_rows = search_multi(
                            [kw1, kw2], re.IGNORECASE, norm_texts
                        )
                        both = sum(
                            1 for r1, r2 in zip(both_rows[0], both_rows[1])
                            if r1 and r2
                        )
                        result["keyword_intersection"] = {
                            "keywords": [kw1, kw2],
                            "both_match": both,
                        }
                        if both > len(positive_ids):
                            result["selectivity_gap"] = (
                                f"Keywords '{kw1}'+'{kw2}' co-occur in "
                                f"{both} docs but your pattern matched "
                                f"only {len(positive_ids)} — "
                                f"{both - len(positive_ids)} docs likely "
                                f"use a different format. Read negatives."
                            )
                    except (RegexSafetyError, RegexTimeoutError):
                        pass

    # Two entity-aware diagnostics using pre-computed keyword doc sets:
    target_set_ids = set(target_ids)
    if (
        ctx.question_entities
        and ctx.keyword_doc_sets
        and not (tiered_decision and tiered_decision.target_was_large)
    ):
        entity = ctx.question_entities[0]
        entity_docs = ctx.keyword_doc_sets.get(entity, set())

        # 1. Scope check: when ALL docs in a narrow subset match, check how
        # many docs OUTSIDE the subset also match the SAME regex pattern.
        # This is the strongest signal: "your pattern matches N more docs
        # outside your working set."
        if (
            len(negative_ids) == 0
            and len(positive_ids) > 0
            and target_set != "all"
        ):
            outside_with_entity = len(entity_docs - target_set_ids)
            if outside_with_entity > 0:
                # Also check: how many of those OUTSIDE docs match the regex?
                outside_pattern_matches = 0
                if compiled_re is not None:
                    for did in entity_docs - target_set_ids:
                        try:
                            text = _normalize_for_regex(
                                ctx.doc_store.get_text(did)
                            )
                            if compiled_re.search(text):
                                outside_pattern_matches += 1
                        except (KeyError, Exception):
                            pass
                if outside_pattern_matches > 0:
                    result["scope_check"] = (
                        f"{outside_pattern_matches} docs outside "
                        f"'{target_set}' also match your pattern."
                    )
                elif outside_with_entity > 0:
                    result["scope_check"] = (
                        f"{outside_with_entity} docs outside set "
                        f"contain '{entity}' (keyword only, not pattern)."
                    )

        # 2. Negative entity check: when negatives exist, count how many
        # contain the entity keyword.
        if negative_ids and positive_ids:
            neg_with_entity = len(entity_docs & set(negative_ids))
            if neg_with_entity > 0:
                result["negative_entity_check"] = (
                    f"{neg_with_entity}/{len(negative_ids)} negatives "
                    f"contain '{entity}' (keyword only)."
                )

    out = ctx.json_dumps(result)

    # Auto-attach snippets centered on regex match.
    # (or doc header for classifier filters).
    if positive_ids:
        out += "\n" + ctx.auto_snippets(positive_ids, n=3, label="positive",
                                         pattern=compiled_re)
    if negative_ids and len(positive_ids) > 0:
        # Center negative snippets on the question's ENTITY keyword (most
        # selective), not the pattern's structural keywords. This shows
        # the agent how the entity appears in docs that the regex MISSED.
        neg_pat = None
        if ctx.question_entities:
            for ent in ctx.question_entities[:3]:
                try:
                    neg_pat = re.compile(re.escape(ent), re.IGNORECASE)
                    break
                except Exception:
                    continue
        if neg_pat is None and filter_type == "regex":
            kws = re.findall(r"[^\W_]{4,}", pattern, re.UNICODE)
            if kws:
                try:
                    neg_pat = re.compile(re.escape(kws[0]), re.IGNORECASE)
                except Exception:
                    neg_pat = None
        out += "\n" + ctx.auto_snippets(
            negative_ids, n=3, label="negative", pattern=neg_pat
        )

    # Context audit: show focused entity context across positive docs.
    if (positive_ids and ctx.question_entities
            and len(positive_ids) >= 3):
        entity = ctx.question_entities[0]
        entity_lower = entity.lower()
        audit_n = min(5, len(positive_ids))
        if len(positive_ids) <= audit_n:
            indices = list(range(len(positive_ids)))
        else:
            indices = sorted(set(
                [round(i * (len(positive_ids) - 1) / (audit_n - 1))
                 for i in range(audit_n)]
            ))
        audit_lines: list[str] = []
        for idx in indices:
            try:
                text = ctx.doc_store.get_text(positive_ids[idx])
                text_lower = text.lower()
                pos = text_lower.find(entity_lower)
                if pos < 0:
                    continue
                win_start = max(0, pos - 40)
                win_end = min(len(text), pos + len(entity) + 40)
                snippet = text[win_start:win_end].replace("\n", " ").strip()
                pfx = "…" if win_start > 0 else ""
                sfx = "…" if win_end < len(text) else ""
                audit_lines.append(f"  {pfx}{snippet}{sfx}")
            except Exception:
                pass
        if audit_lines:
            out += (
                f"\n[Context audit — '{entity}' in "
                f"{len(audit_lines)} of {len(positive_ids)} positive docs:]"
            )
            for line in audit_lines:
                out += "\n" + line

    return out


# -----------------------------------------------------------------------
# Tool: cosine_search
# -----------------------------------------------------------------------


def tool_cosine_search(ctx: ToolContext, args: dict) -> str:
    """Find most similar documents to a query by cosine similarity."""
    query = args.get("query", "")
    top_k = int(args.get("top_k", 10))
    search_set = args.get("search_set", "all")

    if not query:
        return "Error: provide a query."

    candidate_ids = None
    if search_set != "all":
        candidate_ids = ctx.sets.resolve_ids(search_set)

    query_vec = ctx.embedder.embed_single(query)
    results = ctx.embedding_store.cosine_search(
        query_vec, top_k=top_k, candidate_ids=candidate_ids
    )

    output: list[dict] = []
    result_ids: list[str] = []
    for doc_id, score in results:
        result_ids.append(doc_id)
        try:
            meta = ctx.doc_store.get_meta(doc_id)
            output.append(
                {
                    "doc_id": doc_id,
                    "filename": meta.filename,
                    "score": round(score, 4),
                    "chars": meta.char_count,
                }
            )
        except KeyError:
            output.append({"doc_id": doc_id, "score": round(score, 4)})

    set_name = args.get("create_set", "cosine_top_k")
    if result_ids:
        ctx.sets.register(
            set_name,
            result_ids,
            source=f"cosine_search(query={query!r}, top_k={top_k})",
        )

    wrapped = {
        "set_name": set_name,
        "count": len(result_ids),
        "results": output,
    }
    return ctx.json_dumps(wrapped)


# -----------------------------------------------------------------------
# Tool: list_sets
# -----------------------------------------------------------------------


def tool_list_sets(ctx: ToolContext, args: dict) -> str:
    """List all available named document sets with counts."""
    return ctx.json_dumps(ctx.sets.list_sets())


# -----------------------------------------------------------------------
# Tool: list_classifiers
# -----------------------------------------------------------------------


def tool_list_classifiers(ctx: ToolContext, args: dict) -> str:
    """List all trained classifiers."""
    clfs = ctx.classifiers.list_classifiers()
    if not clfs:
        return "No classifiers trained yet."
    return ctx.json_dumps(clfs)


# -----------------------------------------------------------------------
# Tool: search_filenames
# -----------------------------------------------------------------------


def tool_search_filenames(ctx: ToolContext, args: dict) -> str:
    """Search document filenames for a pattern (fast, no LLM call).

    By default does case-insensitive substring match. Set use_regex=true
    for regex matching. Optionally creates a named set.
    Supports target_set to restrict search to an existing set's docs.
    """
    pattern = args.get("pattern", "")
    use_regex = args.get("use_regex", False)
    create_set = args.get("create_set", "")
    target_set = args.get("target_set", "")

    if not pattern:
        return "Error: provide a pattern."

    if not use_regex and re.search(r"[.*+?\[\]()^$|\\]", pattern):
        use_regex = True

    # Resolve target doc IDs if a target_set is specified
    target_ids: set[str] | None = None
    if target_set:
        resolved = ctx.sets.resolve_ids(target_set)
        if not resolved:
            return f"Error: set '{target_set}' is empty or does not exist."
        target_ids = set(resolved)

    matches: list[dict] = []
    indexed_ids: list[str] | None = None
    if ctx.use_tiered_filters and ctx.lexical_index is not None and hasattr(ctx.lexical_index, "search_filenames"):
        terms = ([pattern.replace("_", " ")] if not use_regex else
                 re.findall(r"[^\W_]{3,}", pattern, re.UNICODE)[:6])
        if not terms:
            return "Filename regex refused on a large set: narrow first or include a literal keyword."
        candidates = ctx.lexical_index.search_filenames(terms)
        if candidates.truncated:
            return "Filename search matched too many indexed candidates; add a more selective keyword."
        indexed_ids = candidates.doc_ids
        if target_ids is not None:
            indexed_ids = [did for did in indexed_ids if did in target_ids]

    def _candidate_rows():
        if indexed_ids is None:
            yield from ctx.doc_store.iter_docs()
        else:
            for did in indexed_ids:
                try:
                    yield did, ctx.doc_store.get_meta(did)
                except KeyError:
                    pass

    if use_regex:
        try:
            compile_user_pattern(pattern, re.IGNORECASE)
        except RegexSafetyError as exc:
            return format_safety_error(exc)

        candidate_rows: list[tuple[str, Any]] = []
        for did, meta in _candidate_rows():
            if target_ids is not None and did not in target_ids:
                continue
            candidate_rows.append((did, meta))

        try:
            fn_hits = search_many(
                pattern,
                re.IGNORECASE,
                [meta.filename for _, meta in candidate_rows],
            )
        except RegexTimeoutError as exc:
            return str(exc)

        for (did, meta), hit in zip(candidate_rows, fn_hits):
            if hit:
                matches.append(
                    {"doc_id": did, "filename": meta.filename, "chars": meta.char_count}
                )
    else:
        pat_lower = pattern.lower()
        for did, meta in _candidate_rows():
            if target_ids is not None and did not in target_ids:
                continue
            if pat_lower in meta.filename.lower():
                matches.append(
                    {"doc_id": did, "filename": meta.filename, "chars": meta.char_count}
                )

    searched_count = len(target_ids) if target_ids is not None else ctx.doc_store.doc_count()
    set_created = None
    if create_set and matches:
        matching_ids = [m["doc_id"] for m in matches]
        ctx.sets.register(
            create_set,
            matching_ids,
            source=f"filename_search:{pattern}",
            description=f"Docs matching filename pattern: {pattern}",
        )
        set_created = create_set

    MAX_SHOWN = ctx.filename_max_shown
    if len(matches) <= MAX_SHOWN:
        shown = matches
    else:
        head_n = MAX_SHOWN * 2 // 3
        tail_n = MAX_SHOWN - head_n
        shown = (matches[:head_n]
                 + [{"note": f"... ({len(matches) - MAX_SHOWN} more) ..."}]
                 + matches[-tail_n:])

    result: dict[str, Any] = {
        "count": len(matches),
        "searched": searched_count,
        "matches": shown,
    }
    if target_set:
        result["target_set"] = target_set
    if set_created:
        result["set_created"] = set_created
        result["not_matching"] = searched_count - len(matches)

    if len(matches) > MAX_SHOWN:
        fnames = sorted(m["filename"] for m in matches)
        result["filename_range"] = f"{fnames[0]} ... {fnames[-1]}"

    # When zero matches, show sample filenames so the agent can learn the naming convention
    if not matches:
        sample_fnames: list[str] = []
        source_ids = list(target_ids) if target_ids is not None else []
        if not source_ids:
            source_ids = [did for did, _ in ctx.doc_store.iter_docs()]
        n_samples = min(5, len(source_ids))
        # head + middle + tail sampling
        if n_samples > 0:
            indices = [0, len(source_ids) // 2, len(source_ids) - 1][:n_samples]
            for idx in sorted(set(indices)):
                did = source_ids[idx]
                try:
                    meta = ctx.doc_store.get_meta(did)
                    sample_fnames.append(meta.filename)
                except KeyError:
                    pass
        result["sample_filenames"] = sample_fnames
        result["hint"] = (
            "0 matches. Check the sample filenames above to understand "
            "the naming convention, then adjust your pattern."
        )

    out = ctx.json_dumps(result)

    if matches and set_created:
        match_ids = [m["doc_id"] for m in matches]
        out += "\n" + ctx.auto_snippets(
            match_ids, n=3, label=set_created,
            pattern=ctx.entity_pattern(),
        )

    return out


# -----------------------------------------------------------------------
# Tool: count_set
# -----------------------------------------------------------------------


def tool_count_set(ctx: ToolContext, args: dict) -> str:
    """Count documents in a named set. Shows a sample of doc entries."""
    set_name = args.get("set_name", "all")

    try:
        doc_ids = ctx.sets.resolve_ids(set_name)
    except KeyError as e:
        return str(e)

    def _entry(did: str) -> dict:
        try:
            return {"doc_id": did, "filename": ctx.doc_store.get_meta(did).filename}
        except KeyError:
            return {"doc_id": did, "filename": "[unknown]"}

    max_shown = ctx.count_set_max_shown
    n = len(doc_ids)
    if n <= max_shown:
        shown = [_entry(did) for did in doc_ids]
    else:
        head_n = max_shown * 2 // 3
        tail_n = max_shown - head_n
        head = [_entry(doc_ids[i]) for i in range(head_n)]
        tail = [_entry(doc_ids[i]) for i in range(n - tail_n, n)]
        shown = head + [{"note": f"... ({n - max_shown} more) ..."}] + tail

    result: dict = {"set_name": set_name, "count": n, "docs": shown}

    # Check provenance: if this set was created purely by keyword
    # filters or set operations (without a condition-specific regex),
    # note that the count may include false positives.
    try:
        ds = ctx.sets.get(set_name)
        src = ds.source
        # Sets from apply_filter have source like "regex:..." and end in _pos/_neg.
        # Sets from set_operation have source like "intersect(...)".
        # Implicit expressions (A_and_B) have no registered DocSet.
        is_refined = "_pos" in set_name or "_neg" in set_name
        if not is_refined and n > 0 and set_name != "all":
            result["provenance_note"] = (
                "Unrefined set — count may include false positives."
            )
    except KeyError:
        is_refined = "_pos" in set_name or "_neg" in set_name
        if not is_refined and n > 0 and set_name != "all":
            result["provenance_note"] = (
                "Unrefined set — count may include false positives."
            )

    # Cross-reference with landscape: check if other entity keywords
    # would yield a different count, exposing potential scope gaps.
    if n > 0 and set_name != "all" and ctx.keyword_doc_sets:
        current_set = set(doc_ids)
        for ent in ctx.question_entities[:3]:
            ent_docs = ctx.keyword_doc_sets.get(ent, set())
            outside = len(ent_docs - current_set)
            inside = len(ent_docs & current_set)
            if outside > 0 and inside > 0:
                result["entity_coverage"] = (
                    f"'{ent}': {inside + outside} total docs, "
                    f"{inside} in set, {outside} outside (keyword only)."
                )
                break

    # Filename range: show first and last filenames (sorted) so the
    # agent can see the scope of the set — no arbitrary threshold.
    if n >= 2:
        fnames = sorted(
            e["filename"] for e in (shown if n <= max_shown
                                    else [_entry(d) for d in doc_ids])
            if e.get("filename") and e["filename"] != "[unknown]"
        )
        if len(fnames) >= 2 and fnames[0] != fnames[-1]:
            result["filename_range"] = (
                f"'{fnames[0][:50]}' … '{fnames[-1][:50]}'"
            )

    out = ctx.json_dumps(result)

    # --- Positive vs. Negative contrast ---
    # When counting a _pos set, automatically show entity-centered
    # snippets from BOTH the positive and the corresponding negative
    # set.  This forces the agent to see WHY docs are in/out and
    # makes false positives immediately visible without any domain
    # knowledge.
    neg_name = None
    if set_name.endswith("_pos"):
        neg_name = set_name[:-4] + "_neg"
    elif set_name.endswith("_neg"):
        neg_name = set_name[:-4] + "_pos"
    neg_ids: list[str] = []
    if neg_name:
        try:
            neg_ids = ctx.sets.resolve_ids(neg_name)
        except KeyError:
            neg_ids = []

    entity = ctx.question_entities[0] if ctx.question_entities else None

    def _entity_snippet(did: str, max_chars: int = 120) -> str | None:
        """Short snippet centered on the question entity."""
        try:
            text = ctx.doc_store.get_text(did)
        except KeyError:
            return None
        if not entity:
            snippet = text[:max_chars].replace("\n", " ").strip()
            return f"…{snippet}…" if len(text) > max_chars else snippet
        pos = text.lower().find(entity.lower())
        if pos < 0:
            return None
        start = max(0, pos - 50)
        end = min(len(text), pos + len(entity) + 70)
        snippet = text[start:end].replace("\n", " ").strip()
        pfx = "…" if start > 0 else ""
        sfx = "…" if end < len(text) else ""
        return f"{pfx}{snippet}{sfx}"

    def _sample_indices(total: int, k: int) -> list[int]:
        if total <= k:
            return list(range(total))
        return sorted(set(
            [round(i * (total - 1) / (k - 1)) for i in range(k)]
        ))

    if doc_ids and n > 0:
        pos_lines: list[str] = []
        for idx in _sample_indices(n, 3):
            s = _entity_snippet(doc_ids[idx])
            if s:
                try:
                    fn = ctx.doc_store.get_meta(doc_ids[idx]).filename
                except KeyError:
                    fn = "?"
                pos_lines.append(f"  + [{fn[:50]}] {s}")
        if pos_lines:
            out += "\n[IN set — why these matched:]"
            for line in pos_lines:
                out += "\n" + line

    if neg_ids:
        neg_lines: list[str] = []
        for idx in _sample_indices(len(neg_ids), 2):
            s = _entity_snippet(neg_ids[idx])
            if s:
                try:
                    fn = ctx.doc_store.get_meta(neg_ids[idx]).filename
                except KeyError:
                    fn = "?"
                neg_lines.append(f"  - [{fn[:50]}] {s}")
        if neg_lines:
            out += "\n[OUT of set — why these did NOT match:]"
            for line in neg_lines:
                out += "\n" + line
            out += (
                "\nCompare IN vs OUT snippets above — if they look "
                "similar, the set may contain false positives."
            )

    return out


# -----------------------------------------------------------------------
# Tool: set_operation
# -----------------------------------------------------------------------


def tool_set_operation(ctx: ToolContext, args: dict) -> str:
    """Combine two named sets using set algebra.

    Operations: intersect, union, difference (A minus B).
    Creates a new named result set.
    """
    operation = args.get("operation", "")
    set_a = args.get("set_a", "")
    set_b = args.get("set_b", "")
    result_name = args.get("result_name", "")

    if not operation or not set_a or not set_b:
        return "Error: provide operation ('intersect', 'union', 'difference'), set_a, and set_b."
    if not result_name:
        result_name = f"{set_a}_{operation}_{set_b}"

    try:
        if operation == "intersect":
            ids = ctx.sets.intersect(set_a, set_b)
        elif operation == "union":
            ids = ctx.sets.union(set_a, set_b)
        elif operation == "difference":
            ids = ctx.sets.difference(set_a, set_b)
        else:
            return f"Error: unknown operation {operation!r}. Use 'intersect', 'union', or 'difference'."
    except KeyError as e:
        return str(e)

    ctx.sets.register(
        result_name, ids,
        source=f"{operation}({set_a}, {set_b})",
        description=f"{operation} of {set_a} and {set_b}",
    )

    ids_a = ctx.sets.resolve_ids(set_a)
    ids_b = ctx.sets.resolve_ids(set_b)

    show_limit = ctx.count_set_max_shown if len(ids) > ctx.count_set_max_shown else len(ids)
    samples: list[dict] = []
    for did in ids[:show_limit]:
        try:
            meta = ctx.doc_store.get_meta(did)
            samples.append({"doc_id": did, "filename": meta.filename})
        except KeyError:
            pass

    result_data: dict = {
            "result_set": result_name,
            "count": len(ids),
            "input_counts": {set_a: len(ids_a), set_b: len(ids_b)},
    }
    if len(ids) <= show_limit:
        result_data["all_docs"] = samples
    else:
        result_data["sample_docs"] = samples

    out = ctx.json_dumps(result_data)

    if ids:
        out += "\n" + ctx.auto_snippets(
            ids, n=3, label=result_name,
            pattern=ctx.entity_pattern(),
        )

    # Context audit for set_operation results.
    if ids and ctx.question_entities:
        entity = ctx.question_entities[0]
        entity_lower = entity.lower()
        audit_n = min(5, len(ids))
        if len(ids) <= audit_n:
            indices = list(range(len(ids)))
        else:
            indices = sorted(set(
                [round(i * (len(ids) - 1) / (audit_n - 1))
                 for i in range(audit_n)]
            ))
        audit_lines: list[str] = []
        for idx in indices:
            try:
                text = ctx.doc_store.get_text(ids[idx])
                text_lower = text.lower()
                pos = text_lower.find(entity_lower)
                if pos < 0:
                    continue
                win_start = max(0, pos - 40)
                win_end = min(len(text), pos + len(entity) + 40)
                snippet = text[win_start:win_end].replace("\n", " ").strip()
                prefix = "…" if win_start > 0 else ""
                suffix = "…" if win_end < len(text) else ""
                audit_lines.append(f"  {prefix}{snippet}{suffix}")
            except Exception:
                pass
        if audit_lines:
            out += (
                f"\n[Context audit — '{entity}' in "
                f"{len(audit_lines)} of {len(ids)} docs:]"
            )
            for line in audit_lines:
                out += "\n" + line

    return out


# -----------------------------------------------------------------------
# Tool: explore_patterns
# -----------------------------------------------------------------------


def tool_explore_patterns(ctx: ToolContext, args: dict) -> str:
    """Dry-run multiple regex patterns on a set.  Returns match counts
    and sample filenames for each pattern WITHOUT creating sets.

    The agent can evaluate regex ideas cheaply, then apply_filter the
    best one.  This saves steps especially for weaker models that
    would otherwise trial-and-error one pattern at a time.
    """
    patterns = args.get("patterns", [])
    target_set = args.get("target_set", "all")

    if not patterns or not isinstance(patterns, list):
        return "Error: provide 'patterns' as a list of regex strings (2-5 recommended)."
    if len(patterns) > 8:
        patterns = patterns[:8]

    target_ids = ctx.sets.resolve_ids(target_set)
    if not target_ids:
        return f"Error: target set {target_set!r} is empty."

    refusal = ctx.regex_scan_refusal(target_ids, "Pattern exploration")
    if refusal:
        return f"Filter refused: {refusal}"

    total = len(target_ids)

    # Simpler keyword patterns derived from the agent's regex list.
    keywords: set[str] = set()
    for pat in patterns:
        for word in re.findall(r'[^\W_]{4,}', pat, re.UNICODE):
            keywords.add(word)
    auto_patterns: list[str] = []
    for kw in sorted(keywords)[:3]:
        auto_patterns.append(kw)
    # Deduplicate: don't add auto patterns the agent already included
    existing_lower = {p.lower() for p in patterns}
    auto_patterns = [p for p in auto_patterns if p.lower() not in existing_lower]

    all_patterns = patterns + auto_patterns

    norm_texts: list[str] = []
    for did in target_ids:
        try:
            norm_texts.append(
                _normalize_for_regex(ctx.doc_store.get_text(did))
            )
        except KeyError:
            norm_texts.append("")

    results: list[dict] = []
    safe_patterns: list[str] = []
    for pat in all_patterns:
        try:
            validate_user_pattern(pat)
            safe_patterns.append(pat)
        except RegexSafetyError as exc:
            results.append({"pattern": pat, "error": str(exc)})
        except Exception as exc:
            results.append({"pattern": pat, "error": str(exc)})

    if safe_patterns:
        try:
            hit_matrix = search_multi(safe_patterns, re.IGNORECASE, norm_texts)
        except RegexTimeoutError as exc:
            return str(exc)
    else:
        hit_matrix = []

    for pat, row in zip(safe_patterns, hit_matrix):
        match_ids = [did for did, hit in zip(target_ids, row) if hit]

        sample_fns: list[str] = []
        for did in match_ids[:3]:
            try:
                sample_fns.append(ctx.doc_store.get_meta(did).filename)
            except KeyError:
                pass

        entry: dict = {
            "pattern": pat,
            "matches": len(match_ids),
            "total": total,
            "pct": round(100 * len(match_ids) / total, 1) if total else 0,
        }
        if pat in auto_patterns:
            entry["auto_baseline"] = True
        if sample_fns:
            entry["sample_filenames"] = sample_fns
        if len(match_ids) == 0:
            entry["hint"] = "zero matches — try a simpler pattern"
        results.append(entry)

    # Pairwise coverage for the top two matching patterns.
    # compute exact overlap/union counts and show samples from the regions.
    pairwise: dict | None = None
    ok_only = [r for r in results if "error" not in r and isinstance(r.get("matches"), int)]
    ok_only.sort(key=lambda r: r.get("matches", 0), reverse=True)
    if len(ok_only) >= 2 and ok_only[0].get("matches", 0) > 0:
        a_pat = ok_only[0]["pattern"]
        b_pat = ok_only[1]["pattern"]
        try:
            a_row, b_row = search_multi([a_pat, b_pat], re.IGNORECASE, norm_texts)
            a_only: list[str] = []
            b_only: list[str] = []
            both: list[str] = []
            a_cnt = b_cnt = both_cnt = 0
            for did, a, b in zip(target_ids, a_row, b_row):
                if a:
                    a_cnt += 1
                if b:
                    b_cnt += 1
                if a and b:
                    both_cnt += 1
                    if len(both) < 2:
                        both.append(did)
                elif a and not b:
                    if len(a_only) < 2:
                        a_only.append(did)
                elif b and not a:
                    if len(b_only) < 2:
                        b_only.append(did)

            union_cnt = a_cnt + b_cnt - both_cnt
            pairwise = {
                "patterns": [a_pat, b_pat],
                "counts": {
                    "A": a_cnt,
                    "B": b_cnt,
                    "A_and_B": both_cnt,
                    "A_or_B": union_cnt,
                    "A_only": max(0, a_cnt - both_cnt),
                    "B_only": max(0, b_cnt - both_cnt),
                },
            }
        except (RegexSafetyError, RegexTimeoutError):
            pairwise = None

    out = ctx.json_dumps({
        "target_set": target_set,
        "total_docs": total,
        "pattern_results": results,
        "pairwise_coverage": pairwise,
    })

    # Add snippet from the best-matching pattern showing match context
    best = max((r for r in results if "error" not in r and r.get("matches", 0) > 0),
               key=lambda r: r["matches"], default=None)
    if best and best["matches"] > 0:
        best_pat = best["pattern"]
        try:
            best_hits = search_many(best_pat, re.IGNORECASE, norm_texts)
            sample_ids = [
                did for did, hit in zip(target_ids, best_hits) if hit
            ][:5]
            if sample_ids:
                out += "\n" + ctx.auto_snippets(sample_ids, n=3,
                                                 label=f"best: {best_pat[:40]}",
                                                 pattern=re.compile(best_pat, re.IGNORECASE))
        except Exception:
            pass

    # If we computed pairwise regions, attach samples from A_only / B_only
    # to make format differences explicit without the agent guessing.
    if pairwise and isinstance(pairwise.get("patterns"), list) and len(pairwise["patterns"]) == 2:
        a_pat, b_pat = pairwise["patterns"]
        try:
            a_re = re.compile(a_pat, re.IGNORECASE)
            b_re = re.compile(b_pat, re.IGNORECASE)
            a_only: list[str] = []
            b_only: list[str] = []
            for did in target_ids:
                try:
                    norm = _normalize_for_regex(ctx.doc_store.get_text(did))
                except KeyError:
                    continue
                a = bool(a_re.search(norm))
                b = bool(b_re.search(norm))
                if a and not b and len(a_only) < 2:
                    a_only.append(did)
                if b and not a and len(b_only) < 2:
                    b_only.append(did)
                if len(a_only) >= 2 and len(b_only) >= 2:
                    break
            if a_only:
                out += "\n" + ctx.auto_snippets(a_only, n=len(a_only),
                                                 label=f"A_only: {a_pat[:40]}",
                                                 pattern=a_re)
            if b_only:
                out += "\n" + ctx.auto_snippets(b_only, n=len(b_only),
                                                 label=f"B_only: {b_pat[:40]}",
                                                 pattern=b_re)
        except Exception:
            pass

    return out


# -----------------------------------------------------------------------
# Pattern search (requires ExtractionIndex from preprocessing)
# -----------------------------------------------------------------------


def tool_pattern_search(ctx: ToolContext, args: dict) -> str:
    """Search the pre-computed extraction index with parameters."""
    if not hasattr(ctx, "extraction_index") or ctx.extraction_index is None:
        return "Error: No extraction index available. Run preprocessing first."

    pattern = args.get("pattern", "")
    params = args.get("params", {})
    target_set = args.get("target_set", "")
    max_results = args.get("max_results", 20)

    eidx = ctx.extraction_index
    pi = eidx.get_pattern(pattern)
    if not pi:
        available = eidx.pattern_names
        return ctx.json_dumps({
            "error": f"Unknown pattern: {pattern!r}",
            "available_patterns": available,
        })

    target_ids = None
    if target_set:
        try:
            target_ids = ctx.sets.resolve_ids(target_set)
        except KeyError as e:
            return str(e)

    results = eidx.search(pattern, params, ctx.doc_store, target_ids)
    matched_ids = [r["doc_id"] for r in results]
    matched_set = set(matched_ids)

    set_name = args.get("create_set", "")
    if set_name and matched_ids:
        ctx.sets.register(
            set_name, matched_ids,
            source=f"pattern_search({pattern})",
            description=f"Docs matching {pattern} with {params}",
        )

    # ── Coverage gap analysis (pre-computed at build time) ──────────
    # The extraction index stores per-keyword gap clusters: docs where
    # the keyword appears but NOT in the captured region, grouped by
    # the structural template of the context line.  This lets the
    # agent see distinct structural contexts instantly at inference.
    coverage: dict[str, Any] = {}
    explore_set_name = ""
    if params:
        first_param_val = next(iter(params.values()))
        val_str = (first_param_val if isinstance(first_param_val, str)
                   else str(first_param_val))
        val_lower = val_str.lower()

        target_set_ids = set(target_ids) if target_ids else None

        # Look up pre-computed gap data from extraction index
        precomputed = (pi.keyword_gaps or {}).get(val_lower)

        if precomputed:
            # Use pre-computed gap clusters; track doc_ids per cluster
            all_gap_ids: list[str] = []
            gap_clusters_out: list[dict[str, Any]] = []
            for cluster in precomputed.get("clusters", []):
                cids = cluster["doc_ids"]
                if target_set_ids is not None:
                    cids = [d for d in cids if d in target_set_ids]
                if not cids:
                    continue
                all_gap_ids.extend(cids)
                entry = {
                    "template": cluster["template"],
                    "count": len(cids),
                }
                gap_clusters_out.append(entry)
            gap_ids = sorted(set(all_gap_ids) - matched_set)

            coverage = {
                "confirmed_in_capture": len(matched_ids),
                "keyword_in_pattern_not_captured": len(gap_ids),
            }
            if gap_clusters_out:
                coverage["gap_clusters"] = gap_clusters_out
        else:
            # Fallback: runtime gap detection using keyword doc sets
            pattern_candidates = set(target_ids or pi.matching_doc_ids)
            kw_docs: set[str] | None = None
            for kw, docs in ctx.keyword_doc_sets.items():
                if kw.lower() == val_lower:
                    kw_docs = docs
                    break

            if kw_docs is not None:
                gap_ids = sorted((kw_docs & pattern_candidates) - matched_set)
            elif ctx.regex_scan_refusal(
                sorted(pattern_candidates), "Runtime pattern-gap analysis"
            ) is None:
                gap_ids = []
                for doc_id in sorted(pattern_candidates):
                    if doc_id in matched_set:
                        continue
                    try:
                        text = ctx.doc_store.get_text(doc_id)
                        if val_lower in text.lower():
                            gap_ids.append(doc_id)
                    except KeyError:
                        pass
            else:
                # The indexed pattern result remains valid. Omit the optional
                # coverage diagnostic rather than falling back to a corpus scan.
                gap_ids = []

            coverage = {
                "confirmed_in_capture": len(matched_ids),
                "keyword_in_pattern_not_captured": len(gap_ids),
            }
            # Runtime context extraction for fallback path
            if gap_ids:
                rt_clusters: dict[str, int] = defaultdict(int)
                for doc_id in gap_ids:
                    try:
                        text = ctx.doc_store.get_text(doc_id)
                        idx = text.lower().find(val_lower)
                        if idx >= 0:
                            ls = text.rfind("\n", 0, idx) + 1
                            le = text.find("\n", idx)
                            if le == -1:
                                le = len(text)
                            line = text[ls:le].strip()[:150]
                            tmpl = line.lower().replace(val_lower, "{}")
                            rt_clusters[tmpl] += 1
                    except (KeyError, Exception):
                        pass
                if rt_clusters:
                    coverage["gap_clusters"] = [
                        {"template": t, "count": c}
                        for t, c in sorted(rt_clusters.items(),
                                           key=lambda x: -x[1])
                    ]

        if gap_ids:
            explore_set_name = f"{set_name or pattern}_explore"
            ctx.sets.register(
                explore_set_name, gap_ids,
                source=f"pattern_search({pattern}) coverage gap",
                description=(f"Docs where '{val_str}' appears but not "
                             f"in captured region of {pattern}"),
            )
            coverage["explore_set"] = explore_set_name

    # Capture sample: show what a confirmed match's captured content
    # looks like so the agent can compare with gap docs.
    capture_sample = ""
    if results:
        for r in results[:3]:
            for k, v in r.get("extract", {}).items():
                if v and v.strip():
                    capture_sample = v.strip()[:300]
                    break
            if capture_sample:
                break

    out: dict[str, Any] = {
        "pattern": pattern,
        "description": pi.description,
        "params_used": params,
        "confirmed_matches": len(matched_ids),
    }

    if coverage:
        out["coverage"] = coverage

    if capture_sample:
        out["capture_sample"] = capture_sample

    out["results"] = results[:max_results]

    if set_name:
        out["created_set"] = set_name

    if len(matched_ids) > 0 and not target_set:
        fn_prefix_counts: dict[str, int] = {}
        for doc_id in matched_ids:
            try:
                fn = ctx.doc_store.get_meta(doc_id).filename
                prefix = fn[:4]
                if prefix.isdigit():
                    fn_prefix_counts[prefix] = fn_prefix_counts.get(prefix, 0) + 1
            except (KeyError, AttributeError):
                pass
        if len(fn_prefix_counts) > 1:
            out["filename_year_distribution"] = dict(
                sorted(fn_prefix_counts.items())
            )

    return ctx._truncate(ctx.json_dumps(out))


# -----------------------------------------------------------------------
# Numerical aggregation tools
# -----------------------------------------------------------------------

def tool_extract_field(ctx: ToolContext, args: dict) -> str:
    """Extract a value from each document in a set using a regex capture group."""
    pattern_str = args.get("pattern", "")
    target_set = args.get("target_set", "all")
    group = args.get("group", 1)
    as_type = args.get("as_type", "number")
    save_as = args.get("save_as", "")

    if not pattern_str:
        return "Error: 'pattern' is required (regex with a capture group)."
    if not save_as:
        return "Error: 'save_as' is required (name for the value set)."

    try:
        compile_user_pattern(pattern_str, re.IGNORECASE | re.DOTALL)
    except RegexSafetyError as exc:
        return format_safety_error(exc)

    try:
        doc_ids = ctx.sets.resolve_ids(target_set)
    except KeyError as e:
        return str(e)

    refusal = ctx.regex_scan_refusal(doc_ids, "Field extraction")
    if refusal:
        return f"Filter refused: {refusal}"

    if as_type not in ("number", "text", "date"):
        return "Error: 'as_type' must be one of: number, text, date."

    doc_texts: list[str] = []
    scanned_doc_ids: list[str] = []
    for did in doc_ids:
        try:
            doc_texts.append(_normalize_for_regex(ctx.doc_store.get_text(did)))
            scanned_doc_ids.append(did)
        except KeyError:
            continue
    try:
        raw_groups = search_group_many(
            pattern_str, re.IGNORECASE | re.DOTALL, doc_texts, group
        )
    except RegexTimeoutError as exc:
        return str(exc)

    values: dict[str, float | str] = {}
    date_labels: dict[str, str] = {}
    errors = 0
    for did, raw in zip(scanned_doc_ids, raw_groups):
        if raw is None:
            continue
        raw = raw.strip()
        if as_type == "number":
            cleaned = raw.replace(",", "").replace("$", "").replace("%", "").strip()
            if cleaned.startswith("(") and cleaned.endswith(")"):
                cleaned = "-" + cleaned[1:-1]
            try:
                values[did] = float(cleaned)
            except ValueError:
                errors += 1
        elif as_type == "date":
            try:
                from dateutil import parser as _dateparser
                dt = _dateparser.parse(raw)
                values[did] = dt.timestamp()
                date_labels[did] = dt.strftime("%Y-%m-%d")
            except (ValueError, OverflowError):
                errors += 1
        else:
            values[did] = raw

    ctx.value_sets[save_as] = values
    if date_labels:
        ctx.value_sets[f"{save_as}__labels"] = date_labels

    sample_n = min(8, len(values))
    sample_items = list(values.items())[:sample_n]
    sample = []
    for did, val in sample_items:
        try:
            fn = ctx.doc_store.get_meta(did).filename
        except KeyError:
            fn = did
        display_val = date_labels.get(did, val) if as_type == "date" else val
        sample.append({"doc": fn[:60], "value": display_val})

    result: dict[str, Any] = {
        "value_set": save_as,
        "extracted": len(values),
        "scanned": len(doc_ids),
        "missed": len(doc_ids) - len(values),
        "sample": sample,
    }
    if errors:
        result["parse_errors"] = errors
    if values and as_type == "number":
        nums = [v for v in values.values() if isinstance(v, (int, float))]
        if nums:
            result["preview"] = {
                "min": round(min(nums), 4),
                "max": round(max(nums), 4),
                "mean": round(sum(nums) / len(nums), 4),
            }
    if values and as_type == "date":
        epochs = [v for v in values.values() if isinstance(v, (int, float))]
        if epochs:
            from datetime import datetime, timezone
            result["preview"] = {
                "earliest": datetime.fromtimestamp(min(epochs), tz=timezone.utc).strftime("%Y-%m-%d"),
                "latest": datetime.fromtimestamp(max(epochs), tz=timezone.utc).strftime("%Y-%m-%d"),
                "count": len(epochs),
            }

    return ctx.json_dumps(result)


def tool_extract_list(ctx: ToolContext, args: dict) -> str:
    """Extract ALL matches of a regex from each document (not just the first).

    Unlike extract_field (which returns one scalar per doc), this returns
    a LIST of matches per doc.  Use for repeating structures: analyst
    names, line items, attendee lists, etc.

    The result is a list value set that supports list-aware aggregation:
    unique_values, flatten, count_unique, value_counts, group_by_value.
    """
    pattern_str = args.get("pattern", "")
    target_set = args.get("target_set", "all")
    group = args.get("group", 1)
    save_as = args.get("save_as", "")

    if not pattern_str:
        return "Error: 'pattern' is required (regex with a capture group)."
    if not save_as:
        return "Error: 'save_as' is required (name for the list value set)."

    try:
        compile_user_pattern(pattern_str, re.IGNORECASE | re.MULTILINE)
    except RegexSafetyError as exc:
        return format_safety_error(exc)

    try:
        doc_ids = ctx.sets.resolve_ids(target_set)
    except KeyError as e:
        return str(e)

    refusal = ctx.regex_scan_refusal(doc_ids, "List extraction")
    if refusal:
        return f"Filter refused: {refusal}"

    doc_texts: list[str] = []
    scanned_doc_ids: list[str] = []
    for did in doc_ids:
        try:
            doc_texts.append(ctx.doc_store.get_text(did))
            scanned_doc_ids.append(did)
        except KeyError:
            continue
    try:
        per_doc_items = findall_groups_many(
            pattern_str, re.IGNORECASE | re.MULTILINE, doc_texts, group
        )
    except RegexTimeoutError as exc:
        return str(exc)

    list_values: dict[str, list[str]] = {}
    total_matches = 0
    for did, items in zip(scanned_doc_ids, per_doc_items):
        if items:
            list_values[did] = items
            total_matches += len(items)

    ctx.value_sets[save_as] = list_values  # type: ignore[assignment]
    ctx._list_value_sets.add(save_as)

    all_vals: list[str] = []
    for v in list_values.values():
        all_vals.extend(v)
    unique = sorted(set(all_vals))

    sample_n = min(5, len(list_values))
    sample = []
    for did in list(list_values.keys())[:sample_n]:
        try:
            fn = ctx.doc_store.get_meta(did).filename
        except KeyError:
            fn = did
        vals = list_values[did]
        sample.append({
            "doc": fn[:60],
            "count": len(vals),
            "values": vals[:8] + (["..."] if len(vals) > 8 else []),
        })

    result: dict[str, Any] = {
        "value_set": save_as,
        "type": "list",
        "docs_with_matches": len(list_values),
        "scanned": len(doc_ids),
        "total_matches": total_matches,
        "unique_values": len(unique),
        "sample_docs": sample,
    }
    if len(unique) <= 50:
        result["all_unique_values"] = unique
    else:
        result["unique_sample"] = unique[:25] + ["..."] + unique[-10:]

    return ctx.json_dumps(result)


def _aggregate_list(
    ctx: ToolContext, vs_name: str, vset: dict, operation: str, create_set: str,
) -> str:
    """Aggregate a list value set (from extract_list)."""
    LIST_OPS = (
        "count", "count_unique", "unique_values",
        "flatten", "value_counts", "group_by_value",
        "per_doc_count", "per_doc_count_unique",
    )
    if operation not in LIST_OPS:
        return (f"Error: for list value sets, 'operation' must be one of: "
                f"{', '.join(LIST_OPS)}.")

    all_items: list[str] = []
    per_doc_counts: dict[str, int] = {}
    for did, vals in vset.items():
        if isinstance(vals, list):
            all_items.extend(vals)
            per_doc_counts[did] = len(vals)

    if operation == "count":
        return ctx.json_dumps({
            "value_set": vs_name, "operation": "count",
            "result": len(all_items),
            "n_docs": len(per_doc_counts),
        })

    if operation == "count_unique":
        unique = set(v.lower() if isinstance(v, str) else v for v in all_items)
        return ctx.json_dumps({
            "value_set": vs_name, "operation": "count_unique",
            "result": len(unique),
            "total_occurrences": len(all_items),
            "n_docs": len(per_doc_counts),
        })

    if operation == "unique_values":
        seen_lower: dict[str, str] = {}
        for v in all_items:
            key = v.lower() if isinstance(v, str) else str(v)
            if key not in seen_lower:
                seen_lower[key] = v
        unique = sorted(seen_lower.values(),
                        key=lambda x: x.lower() if isinstance(x, str) else str(x))
        result: dict[str, Any] = {
            "value_set": vs_name, "operation": "unique_values",
            "count": len(unique),
        }
        if len(unique) <= 100:
            result["values"] = unique
        else:
            result["values"] = unique[:50] + [f"... ({len(unique) - 60} more)"] + unique[-10:]
        return ctx.json_dumps(result)

    if operation == "flatten":
        result = {
            "value_set": vs_name, "operation": "flatten",
            "total": len(all_items),
            "n_docs": len(per_doc_counts),
        }
        if len(all_items) <= 200:
            result["values"] = all_items
        else:
            result["values"] = all_items[:100] + [f"... ({len(all_items) - 110} more)"] + all_items[-10:]
        return ctx.json_dumps(result)

    if operation == "value_counts":
        from collections import Counter
        counter: Counter = Counter()
        for v in all_items:
            counter[v.lower() if isinstance(v, str) else v] += 1
        ranked = counter.most_common(50)
        result = {
            "value_set": vs_name, "operation": "value_counts",
            "unique_count": len(counter),
            "top_values": [{"value": v, "count": c} for v, c in ranked],
        }
        if len(counter) > 50:
            result["note"] = f"Showing top 50 of {len(counter)} unique values."
        if create_set:
            top_val = ranked[0][0] if ranked else ""
            matching_docs = [
                did for did, vals in vset.items()
                if isinstance(vals, list) and any(
                    (v.lower() if isinstance(v, str) else v) == top_val
                    for v in vals
                )
            ]
            if matching_docs:
                ctx.sets.register(
                    create_set, matching_docs,
                    source=f"value_counts:top({vs_name})",
                    description=f"Docs containing '{top_val}' from {vs_name}",
                )
                result["created_set"] = create_set
        return ctx.json_dumps(result)

    if operation == "group_by_value":
        groups: dict[str, list[str]] = defaultdict(list)
        for did, vals in vset.items():
            if isinstance(vals, list):
                for v in vals:
                    key = v.lower() if isinstance(v, str) else str(v)
                    groups[key].append(did)
        ranked = sorted(groups.items(), key=lambda x: -len(x[1]))
        result = {
            "value_set": vs_name, "operation": "group_by_value",
            "unique_count": len(groups),
        }
        entries = []
        for val, dids in ranked[:40]:
            try:
                fns = [ctx.doc_store.get_meta(d).filename[:50] for d in dids[:5]]
            except KeyError:
                fns = dids[:5]
            entries.append({
                "value": val, "doc_count": len(dids),
                "sample_docs": fns,
            })
        result["groups"] = entries
        if len(groups) > 40:
            result["note"] = f"Showing top 40 of {len(groups)} groups."
        return ctx.json_dumps(result)

    if operation in ("per_doc_count", "per_doc_count_unique"):
        scalar_vs: dict[str, float] = {}
        for did, vals in vset.items():
            if not isinstance(vals, list):
                continue
            if operation == "per_doc_count":
                scalar_vs[did] = float(len(vals))
            else:
                scalar_vs[did] = float(len(set(
                    v.lower() if isinstance(v, str) else v for v in vals
                )))

        save_name = create_set or f"{vs_name}_{operation}"
        ctx.value_sets[save_name] = scalar_vs

        vals_list = sorted(scalar_vs.values())
        n = len(vals_list)
        result = {
            "value_set": vs_name,
            "operation": operation,
            "created_value_set": save_name,
            "n_docs": n,
        }
        if n > 0:
            avg = sum(vals_list) / n
            result["min"] = vals_list[0]
            result["max"] = vals_list[-1]
            result["avg"] = round(avg, 2)
            result["median"] = vals_list[n // 2]
            sample_n = min(8, n)
            sample = []
            for did in list(scalar_vs.keys())[:sample_n]:
                try:
                    fn = ctx.doc_store.get_meta(did).filename[:50]
                except KeyError:
                    fn = did
                sample.append({"doc": fn, "count": int(scalar_vs[did])})
            result["sample"] = sample
        return ctx.json_dumps(result)

    return f"Unknown list operation: {operation}"


def tool_aggregate(ctx: ToolContext, args: dict) -> str:
    """Compute an aggregate over extracted values.

    Scalar value sets (from extract_field): sum, avg, min, max, count.
    List value sets (from extract_list): unique_values, count_unique,
    flatten, value_counts, group_by_value.
    """
    vs_name = args.get("value_set", "")
    operation = args.get("operation", "").lower()
    create_set = args.get("create_set", "")

    if not vs_name:
        return "Error: 'value_set' is required."
    if vs_name not in ctx.value_sets:
        available = list(ctx.value_sets.keys()) or ["(none)"]
        return f"Unknown value set '{vs_name}'. Available: {available}"

    vset = ctx.value_sets[vs_name]

    # Detect list value sets (from extract_list)
    is_list_vs = (
        vs_name in ctx._list_value_sets
        or any(isinstance(v, list) for v in vset.values())
    )
    if is_list_vs:
        return _aggregate_list(ctx, vs_name, vset, operation, create_set)

    SCALAR_OPS = ("sum", "avg", "min", "max", "count")
    if operation not in SCALAR_OPS:
        return (f"Error: for scalar value sets, 'operation' must be one of: "
                f"{', '.join(SCALAR_OPS)}.")

    nums = {did: v for did, v in vset.items() if isinstance(v, (int, float))}

    if operation == "count":
        result: dict[str, Any] = {
            "value_set": vs_name,
            "operation": "count",
            "result": len(vset),
        }
        return ctx.json_dumps(result)

    if not nums:
        return f"No numeric values in value set '{vs_name}' (has {len(vset)} text entries)."

    vals = list(nums.values())
    if operation == "sum":
        agg = sum(vals)
    elif operation == "avg":
        agg = sum(vals) / len(vals)
    elif operation == "min":
        agg = min(vals)
    elif operation == "max":
        agg = max(vals)
    else:
        return f"Unknown operation: {operation}"

    label_key = f"{vs_name}__labels"
    labels = ctx.value_sets.get(label_key, {})

    def _display(val: float, doc_id: str = "") -> str | float:
        if labels and doc_id:
            return labels.get(doc_id, val)
        if labels:
            from datetime import datetime, timezone
            return datetime.fromtimestamp(val, tz=timezone.utc).strftime("%Y-%m-%d")
        return val

    result = {
        "value_set": vs_name,
        "operation": operation,
        "result": _display(agg),
        "n_values": len(vals),
    }

    if operation in ("min", "max"):
        target_docs = [did for did, v in nums.items() if v == agg]
        doc_entries = []
        for did in target_docs[:5]:
            try:
                fn = ctx.doc_store.get_meta(did).filename
            except KeyError:
                fn = did
            doc_entries.append({"doc_id": did, "filename": fn, "value": _display(agg, did)})
        result["matching_docs"] = doc_entries
        if create_set and target_docs:
            ctx.sets.register(
                create_set, target_docs,
                source=f"aggregate:{operation}({vs_name})",
                description=f"Doc(s) with {operation} value from '{vs_name}'",
            )
            result["created_set"] = create_set

    return ctx.json_dumps(result)


def tool_extract_from_filename(ctx: ToolContext, args: dict) -> str:
    """Extract a value from each document's FILENAME using a regex capture group.

    Unlike extract_field (which runs on document text), this runs the regex
    on the filename.  Use for metadata encoded in filenames: company tickers,
    dates, quarters, categories, etc.

    Stores results as a scalar value set (doc → captured string).
    """
    pattern_str = args.get("pattern", "")
    target_set = args.get("target_set", "all")
    save_as = args.get("save_as", "")
    group = args.get("group", 1)

    if not pattern_str:
        return "Error: 'pattern' is required (regex with a capture group)."
    if not save_as:
        return "Error: 'save_as' is required."

    try:
        compile_user_pattern(pattern_str, re.IGNORECASE)
    except RegexSafetyError as exc:
        return format_safety_error(exc)

    try:
        doc_ids = ctx.sets.resolve_ids(target_set)
    except KeyError as e:
        return str(e)

    filenames: list[str] = []
    valid_ids: list[str] = []
    for did in doc_ids:
        try:
            filenames.append(ctx.doc_store.get_meta(did).filename)
            valid_ids.append(did)
        except KeyError:
            continue

    try:
        raw_groups = search_group_many(
            pattern_str, re.IGNORECASE, filenames, group
        )
    except RegexTimeoutError as exc:
        return str(exc)

    values: dict[str, str] = {}
    errors = 0
    for did, raw in zip(valid_ids, raw_groups):
        if raw is None:
            continue
        val = raw.strip()
        if val:
            values[did] = val
        else:
            errors += 1

    ctx.value_sets[save_as] = values  # type: ignore[assignment]

    unique = sorted(set(values.values()))
    sample_n = min(8, len(values))
    sample = []
    for did in list(values.keys())[:sample_n]:
        try:
            fn = ctx.doc_store.get_meta(did).filename[:60]
        except KeyError:
            fn = did
        sample.append({"doc": fn, "value": values[did]})

    result: dict[str, Any] = {
        "value_set": save_as,
        "extracted": len(values),
        "scanned": len(doc_ids),
        "unique_values": len(unique),
        "sample": sample,
    }
    if len(unique) <= 50:
        result["all_unique"] = unique
    if errors:
        result["errors"] = errors
    return ctx.json_dumps(result)


def tool_filter_values(ctx: ToolContext, args: dict) -> str:
    """Filter a value set by a comparison threshold and create a doc set."""
    vs_name = args.get("value_set", "")
    op = args.get("operator", "").strip()
    threshold_raw = args.get("threshold")
    create_set = args.get("create_set", "")

    if not vs_name:
        return "Error: 'value_set' is required."
    if vs_name not in ctx.value_sets:
        available = list(ctx.value_sets.keys()) or ["(none)"]
        return f"Unknown value set '{vs_name}'. Available: {available}"
    if not create_set:
        return "Error: 'create_set' is required (name for the resulting doc set)."

    OPS = {
        "<": lambda v, t: v < t, "lt": lambda v, t: v < t,
        "<=": lambda v, t: v <= t, "le": lambda v, t: v <= t,
        ">": lambda v, t: v > t, "gt": lambda v, t: v > t,
        ">=": lambda v, t: v >= t, "ge": lambda v, t: v >= t,
        "==": lambda v, t: v == t, "eq": lambda v, t: v == t,
        "!=": lambda v, t: v != t, "ne": lambda v, t: v != t,
    }
    if op not in OPS:
        return f"Error: 'operator' must be one of: <, <=, >, >=, ==, != (or lt/le/gt/ge/eq/ne). Got '{op}'."

    vset = ctx.value_sets[vs_name]
    label_key = f"{vs_name}__labels"
    is_date = label_key in ctx.value_sets

    if threshold_raw is None:
        return "Error: 'threshold' is required."

    if is_date:
        if isinstance(threshold_raw, str):
            try:
                from dateutil import parser as _dateparser
                threshold = _dateparser.parse(threshold_raw).timestamp()
            except (ValueError, OverflowError):
                return f"Cannot parse threshold '{threshold_raw}' as a date."
        else:
            threshold = float(threshold_raw)
    else:
        try:
            threshold = float(threshold_raw)
        except (ValueError, TypeError):
            return f"Cannot parse threshold '{threshold_raw}' as a number."

    cmp = OPS[op]
    matching = [did for did, v in vset.items()
                if isinstance(v, (int, float)) and cmp(v, threshold)]

    ctx.sets.register(
        create_set, matching,
        source=f"filter_values:{vs_name} {op} {threshold_raw}",
        description=f"Docs where {vs_name} {op} {threshold_raw}",
    )

    result: dict[str, Any] = {
        "created_set": create_set,
        "matched": len(matching),
        "total_in_value_set": len(vset),
    }
    if matching:
        sample_items = matching[:5]
        sample = []
        for did in sample_items:
            try:
                fn = ctx.doc_store.get_meta(did).filename
            except KeyError:
                fn = did
            val = vset[did]
            if is_date:
                labels = ctx.value_sets.get(label_key, {})
                val = labels.get(did, val)
            sample.append({"doc": fn[:60], "value": val})
        result["sample"] = sample

    return ctx.json_dumps(result)


# -----------------------------------------------------------------------
# Tool dispatch
# -----------------------------------------------------------------------

# ── calculator ──────────────────────────────────────────────────────

import ast
import math
import operator

_CALC_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_CALC_FUNCS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "sqrt": math.sqrt,
    "cbrt": lambda x: math.copysign(abs(x) ** (1 / 3), x),
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "ln": math.log,
    "exp": math.exp,
    "ceil": math.ceil,
    "floor": math.floor,
    "pow": math.pow,
    "factorial": math.factorial,
    "gcd": math.gcd,
}

_CALC_CONSTANTS: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "inf": math.inf,
}

_MAX_EXPONENT = 1000


def _safe_eval_expr(node: ast.expr, variables: dict[str, float]) -> float | int:
    """Recursively evaluate an AST expression node with whitelisted operations."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")

    if isinstance(node, ast.Name):
        name = node.id
        if name in variables:
            return variables[name]
        if name in _CALC_CONSTANTS:
            return _CALC_CONSTANTS[name]
        raise ValueError(f"Unknown variable: '{name}'")

    if isinstance(node, ast.UnaryOp):
        op = _CALC_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported unary op: {type(node.op).__name__}")
        return op(_safe_eval_expr(node.operand, variables))

    if isinstance(node, ast.BinOp):
        left = _safe_eval_expr(node.left, variables)
        right = _safe_eval_expr(node.right, variables)
        op = _CALC_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported binary op: {type(node.op).__name__}")
        if isinstance(node.op, ast.Pow):
            if isinstance(right, (int, float)) and abs(right) > _MAX_EXPONENT:
                raise ValueError(f"Exponent too large: {right} (max {_MAX_EXPONENT})")
        return op(left, right)

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls allowed (e.g. sqrt(x))")
        fname = node.func.id
        if fname not in _CALC_FUNCS:
            raise ValueError(
                f"Unknown function: '{fname}'. Available: {sorted(_CALC_FUNCS)}"
            )
        eval_args = [_safe_eval_expr(a, variables) for a in node.args]
        return _CALC_FUNCS[fname](*eval_args)

    if isinstance(node, ast.List):
        return [_safe_eval_expr(el, variables) for el in node.elts]  # type: ignore[return-value]

    if isinstance(node, ast.Tuple):
        return tuple(_safe_eval_expr(el, variables) for el in node.elts)  # type: ignore[return-value]

    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def _calc_evaluate(expr: str, variables: dict[str, float] | None = None) -> float | int:
    """Safely evaluate a mathematical expression string."""
    tree = ast.parse(expr.strip(), mode="eval")
    return _safe_eval_expr(tree.body, variables or {})


def tool_calculate(ctx: ToolContext, args: dict) -> str:
    """Evaluate mathematical expressions safely.

    Supports: +, -, *, /, //, %, ** (exponents incl. fractional),
    parentheses, and functions: sqrt, cbrt, log, log2, log10, ln, exp,
    abs, round, ceil, floor, pow, min, max, sum, factorial, gcd.
    Constants: pi, e.

    Can evaluate a single expression or multiple named expressions
    that build on each other (variables from earlier steps are
    available in later steps).
    """
    expressions = args.get("expressions")
    expr_single = args.get("expr", "")

    if not expressions and not expr_single:
        return "Error: provide 'expr' (single expression) or 'expressions' (list of {\"name\": ..., \"expr\": ...})."

    variables: dict[str, float] = {}
    for k, v in args.get("variables", {}).items():
        if isinstance(v, (int, float)):
            variables[k] = v

    results: list[dict[str, Any]] = []

    if expr_single and not expressions:
        expressions = [{"name": "result", "expr": expr_single}]

    for step in expressions:
        name = step.get("name", f"step_{len(results)}")
        expr = step.get("expr", "")
        if not expr:
            results.append({"name": name, "error": "empty expression"})
            continue
        try:
            value = _calc_evaluate(expr, variables)
            if isinstance(value, float):
                if math.isnan(value):
                    results.append({"name": name, "expr": expr, "error": "result is NaN"})
                    continue
                if math.isinf(value):
                    results.append({"name": name, "expr": expr, "error": "result is infinite"})
                    continue
            variables[name] = value  # type: ignore[assignment]
            display = f"{value:,.6f}".rstrip("0").rstrip(".") if isinstance(value, float) else str(value)
            results.append({"name": name, "expr": expr, "value": value, "display": display})
        except Exception as e:
            results.append({"name": name, "expr": expr, "error": str(e)})

    if len(results) == 1:
        return ctx.json_dumps(results[0])
    return ctx.json_dumps({"steps": results, "final": results[-1]})


PATTERN_TOOLS = frozenset({"pattern_search"})

TOOL_REGISTRY: dict[str, tuple] = {
    "read_docs": (tool_read_docs, "Read documents by ID or set name"),
    "find_in_doc": (tool_find_in_doc, "Find occurrences inside a doc (char offsets for paging)"),
    "table_lookup": (tool_table_lookup, "Locate a table row and return header context (structured, offsets)"),
    "label_docs": (tool_label_docs, "Read and label docs for conditions (0/1)"),
    "create_regex": (tool_create_regex, "Create or register a regex pattern"),
    "train_classifier": (tool_train_classifier, "Train a binary classifier"),
    "apply_filter": (tool_apply_filter, "Apply regex or classifier to a set"),
    "cosine_search": (tool_cosine_search, "Semantic similarity search"),
    "list_sets": (tool_list_sets, "List all named document sets"),
    "list_classifiers": (tool_list_classifiers, "List trained classifiers"),
    "search_filenames": (tool_search_filenames, "Search filenames for a pattern"),
    "count_set": (tool_count_set, "Count docs in a set"),
    "set_operation": (tool_set_operation, "Intersect / union / difference of two sets"),
    "explore_patterns": (tool_explore_patterns, "Dry-run multiple regex patterns, see match stats"),
    "pattern_search": (tool_pattern_search, "Search pre-computed extraction index with parameters"),
    "extract_field": (tool_extract_field, "Extract a value from each doc using a regex capture group"),
    "extract_list": (tool_extract_list, "Extract ALL matches of a regex from each doc as a list"),
    "aggregate": (tool_aggregate, "Compute sum/avg/min/max/count over extracted values, or unique_values/count_unique/value_counts/group_by_value for list sets"),
    "filter_values": (tool_filter_values, "Filter a value set by threshold, create a doc set"),
    "extract_from_filename": (tool_extract_from_filename, "Extract metadata from filenames via regex (company, quarter, etc.)"),
    "calculate": (tool_calculate, "Evaluate math expressions: arithmetic, exponents, roots, log, trig, multi-step"),
}

EMBEDDING_TOOLS = frozenset({
    "cosine_search", "train_classifier", "list_classifiers",
    # Labeling is primarily used to build classifiers; hide it when embeddings are off.
    "label_docs",
})

AGG_TOOLS = frozenset({
    "extract_field", "extract_list", "aggregate", "filter_values",
    "extract_from_filename",
})


def get_tool_registry(
    use_embeddings: bool = True,
    disabled_tools: frozenset[str] | None = None,
    use_patterns: bool = False,
    use_agg_tools: bool = False,
) -> dict[str, tuple]:
    """Return the tool registry, excluding embedding-dependent or explicitly disabled tools."""
    exclude = set()
    if not use_embeddings:
        exclude |= EMBEDDING_TOOLS
    if not use_patterns:
        exclude |= PATTERN_TOOLS
    if not use_agg_tools:
        exclude |= AGG_TOOLS
    if disabled_tools:
        exclude |= set(disabled_tools)
    if not exclude:
        return TOOL_REGISTRY
    return {k: v for k, v in TOOL_REGISTRY.items() if k not in exclude}


def execute_tool(ctx: ToolContext, tool_name: str, args: dict) -> str:
    """Dispatch a tool call, optionally appending alternative outcomes."""
    registry = get_tool_registry(
        ctx.use_embeddings, ctx.disabled_tools,
        use_patterns=ctx.extraction_index is not None,
        use_agg_tools=ctx.use_agg_tools,
    )
    if tool_name not in registry:
        return f"Unknown tool: {tool_name}. Available: {list(registry.keys())}"
    fn, _ = registry[tool_name]
    try:
        result = fn(ctx, args)
    except Exception as e:
        return f"Tool {tool_name} error: {type(e).__name__}: {e}"

    try:
        alt = generate_alternatives(ctx, tool_name, args, result)
        if alt:
            result += alt
    except Exception:
        pass

    # Safety net: no single tool result should exceed the per-tool budget
    if len(result) > ctx.max_tool_result_chars:
        result = (
            result[: ctx.max_tool_result_chars]
            + f"\n[... truncated at {ctx.max_tool_result_chars} chars]"
        )

    return result
