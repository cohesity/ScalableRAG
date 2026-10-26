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

"""Optional suffixes appended to tool results after set-creating calls.

Computes set-operation previews (intersect/union/difference of recent sets)
and filename match counts after regex filters. O(N) in document count; no
full-text reads and no LLM calls.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tools import ToolContext


def generate_alternatives(
    ctx: "ToolContext",
    tool_name: str,
    args: dict,
    result: str,
) -> str:
    """Append useful planning info to tool results.

    Returns a string to append, or empty string if nothing useful.
    """
    parts: list[str] = []

    if tool_name == "apply_filter" and args.get("filter_type") == "regex":
        fn_info = _filename_search_counts(ctx, args)
        if fn_info:
            parts.append(fn_info)

    if (tool_name in ("apply_filter", "set_operation", "search_filenames")
            and "set_operation" not in ctx.disabled_tools):
        comp = _available_compositions(ctx)
        if comp:
            parts.append(comp)

    if not parts:
        return ""

    header = "\n\n═══ Automatically computed ═══"
    return header + "\n" + "\n".join(parts)


def _filename_search_counts(ctx: "ToolContext", args: dict) -> str:
    """After regex filter, show filename-match counts for key terms."""
    pattern = args.get("pattern", "")
    if not pattern:
        return ""

    words = re.findall(r"[^\W_]{4,}", pattern, re.UNICODE)
    seen: set[str] = set()
    unique_words: list[str] = []
    for w in words:
        wl = w.lower()
        if wl not in seen:
            seen.add(wl)
            unique_words.append(w)

    if not unique_words:
        return ""

    fn_results: list[tuple[str, int]] = []
    for word in unique_words[:3]:
        if ctx.use_tiered_filters and ctx.lexical_index is not None and hasattr(ctx.lexical_index, "count_filenames"):
            count = ctx.lexical_index.count_filenames(word)
            if count > 0:
                fn_results.append((word, count))
            continue
        count = 0
        wl = word.lower()
        for _, meta in ctx.doc_store.iter_docs():
            if wl in meta.filename.lower():
                count += 1
        if count > 0:
            fn_results.append((word, count))

    if not fn_results:
        return ""

    lines = ["Filename search counts:"]
    for word, count in fn_results:
        lines.append(f"  search_filenames(\"{word}\") → {count} docs")
    return "\n".join(lines)


_PREREGISTERED_PREFIXES = ("preprocessing:", "filename_category:", "corpus")


def _available_compositions(ctx: "ToolContext") -> str:
    """If 2+ agent-created sets exist, show composition counts.

    Skips pre-registered sets (cat_*, pat_*) which the agent didn't
    create, to avoid confusing cross-comparisons.
    """
    set_names = [
        name for name in ctx.sets._sets
        if (name != "all"
            and not name.endswith("_neg")
            and not ctx.sets._sets[name].source.startswith(_PREREGISTERED_PREFIXES))
    ]

    if len(set_names) < 2:
        return ""

    recent = set_names[-2:]
    a_name, b_name = recent[0], recent[1]
    a_ids = set(ctx.sets.resolve_ids(a_name))
    b_ids = set(ctx.sets.resolve_ids(b_name))

    inter = len(a_ids & b_ids)
    union_count = len(a_ids | b_ids)
    diff_ab = len(a_ids - b_ids)
    diff_ba = len(b_ids - a_ids)

    if inter == len(a_ids) and inter == len(b_ids):
        return ""

    lines = [
        f"Set compositions — {a_name} ({len(a_ids)}) × {b_name} ({len(b_ids)}):",
        f"  intersect → {inter}",
        f"  union     → {union_count}",
        f"  {a_name} \\ {b_name} → {diff_ab}",
        f"  {b_name} \\ {a_name} → {diff_ba}",
    ]
    return "\n".join(lines)
