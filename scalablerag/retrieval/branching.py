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

"""Branch-and-Select: preview deterministic tool outcomes and let the LLM pick.

Most retrieval tools (regex filters, set ops, filename search) are deterministic.
This module generates alternative tool calls, previews each outcome without
mutating state, and formats the options for a selection step.

Steps:
1. Take the agent's proposed tool call
2. Generate K alternative actions (heuristics, no LLM)
3. Preview all K+1 outcomes
4. Format them for LLM selection
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from .tiered_filter import FilterBudgetExceeded, TieredRegexFilter

if TYPE_CHECKING:
    from .tools import ToolContext

DETERMINISTIC_TOOLS = frozenset({
    "apply_filter", "search_filenames", "set_operation", "count_set",
    "extract_field", "aggregate",
    "find_in_doc", "table_lookup",
})


def preview_apply_filter(ctx: "ToolContext", pattern: str, target_set: str) -> dict:
    """Count regex matches without creating named sets."""
    try:
        compiled = re.compile(pattern, re.IGNORECASE)
    except re.error:
        return {"error": f"invalid regex: {pattern}", "count": -1}

    try:
        target_ids = ctx.sets.resolve_ids(target_set)
    except KeyError as e:
        return {"error": str(e), "count": -1}
    if ctx.use_tiered_filters and ctx.lexical_index is not None:
        try:
            decision = TieredRegexFilter(
                ctx.doc_store, ctx.lexical_index, ctx.tier_filter_config
            ).apply(pattern, target_ids)
        except FilterBudgetExceeded as exc:
            return {"error": str(exc), "count": -1, "total": len(target_ids)}
        return {
            "count": len(decision.positive_ids),
            "total": len(target_ids),
            "samples": [],
        }

    hits = 0
    sample_fnames: list[str] = []
    if hasattr(ctx.doc_store, "iter_normalized_texts"):
        rows = ctx.doc_store.iter_normalized_texts(target_ids)
    elif hasattr(ctx.doc_store, "iter_texts"):
        rows = ctx.doc_store.iter_texts(target_ids)
    else:
        def _rows():
            for doc_id in target_ids:
                try:
                    yield doc_id, ctx.doc_store.get_text(doc_id)
                except KeyError:
                    pass
        rows = _rows()
    for did, text in rows:
        if compiled.search(text):
            hits += 1
            if len(sample_fnames) < 3:
                meta = ctx.doc_store.get_meta(did)
                sample_fnames.append(meta.filename)

    return {
        "count": hits,
        "total": len(target_ids),
        "samples": sample_fnames,
    }


def preview_search_filenames(ctx: "ToolContext", pattern: str,
                             use_regex: bool = False) -> dict:
    """Count filename matches without creating named sets."""
    # Auto-detect regex (same logic as the actual tool)
    if not use_regex and re.search(r"[.*+?\[\]()^$|\\]", pattern):
        use_regex = True

    hits = 0
    sample_fnames: list[str] = []
    rows = None
    if ctx.use_tiered_filters and ctx.lexical_index is not None and hasattr(ctx.lexical_index, "search_filenames"):
        terms = ([pattern.replace("_", " ")] if not use_regex else
                 re.findall(r"[^\W_]{3,}", pattern, re.UNICODE)[:6])
        if terms:
            found = ctx.lexical_index.search_filenames(terms)
            if not found.truncated:
                rows = []
                for did in found.doc_ids:
                    try: rows.append((did, ctx.doc_store.get_meta(did)))
                    except KeyError: pass

    if use_regex:
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
        except re.error:
            return {"error": f"invalid regex: {pattern}", "count": -1}
        for did, meta in (rows if rows is not None else ctx.doc_store.iter_docs()):
            if compiled.search(meta.filename):
                hits += 1
                if len(sample_fnames) < 3:
                    sample_fnames.append(meta.filename)
    else:
        pat_lower = pattern.lower()
        for did, meta in (rows if rows is not None else ctx.doc_store.iter_docs()):
            if pat_lower in meta.filename.lower():
                hits += 1
                if len(sample_fnames) < 3:
                    sample_fnames.append(meta.filename)

    return {
        "count": hits,
        "total": ctx.doc_store.doc_count(),
        "samples": sample_fnames,
    }


def preview_set_operation(ctx: "ToolContext", operation: str,
                          set_a: str, set_b: str) -> dict:
    """Compute set operation result size without creating a named set."""
    try:
        ids_a = set(ctx.sets.resolve_ids(set_a))
        ids_b = set(ctx.sets.resolve_ids(set_b))
    except KeyError as e:
        return {"error": str(e), "count": -1}

    if "intersect" in operation:
        result = ids_a & ids_b
    elif "union" in operation:
        result = ids_a | ids_b
    elif "diff" in operation:
        result = ids_a - ids_b
    else:
        return {"error": f"unknown operation: {operation}", "count": -1}

    return {"count": len(result), "a_size": len(ids_a), "b_size": len(ids_b)}


def preview_action(ctx: "ToolContext", tool: str, args: dict) -> dict:
    """Preview a single action's outcome without state modification."""
    if tool == "apply_filter" and args.get("filter_type") == "regex":
        pattern = args.get("pattern", "")
        target = args.get("target_set", "all")
        return preview_apply_filter(ctx, pattern, target)

    if tool == "search_filenames":
        return preview_search_filenames(
            ctx, args.get("pattern", ""),
            use_regex=args.get("use_regex", False),
        )

    if tool == "set_operation":
        return preview_set_operation(
            ctx, args.get("operation", ""),
            args.get("set_a", ""), args.get("set_b", ""),
        )

    if tool == "count_set":
        try:
            ids = ctx.sets.resolve_ids(args.get("set_name", "all"))
            return {"count": len(ids)}
        except KeyError as e:
            return {"error": str(e), "count": -1}

    if tool in {"find_in_doc", "table_lookup"}:
        raw = args.get("doc_id") or ""
        raw_ids = args.get("doc_ids") or []
        set_name = args.get("set_name") or ""
        if raw and not raw_ids:
            raw_ids = [raw]
        if set_name and not raw_ids:
            try:
                ids = ctx.sets.resolve_ids(set_name)
            except KeyError as e:
                return {"error": str(e), "count": -1}
            raw_ids = [ids[0]] if ids else []
        if not raw_ids or len(raw_ids) != 1:
            return {"error": "need exactly one doc_id", "count": -1}

        did = ctx.doc_store.resolve_doc_id(raw_ids[0]) or raw_ids[0]
        try:
            text = ctx.doc_store.get_text(did)
        except KeyError:
            return {"error": f"doc not found: {raw_ids[0]!r}", "count": -1}

        # Apply range restriction
        n = len(text)
        try:
            s = int(args.get("start_char", 0))
        except Exception:
            s = 0
        s = max(0, min(n, s))
        e_raw = args.get("end_char", None)
        if e_raw is None:
            e = n
        else:
            try:
                e = int(e_raw)
            except Exception:
                e = n
            e = max(s, min(n, e))
        segment = text[s:e]

        if tool == "find_in_doc":
            pattern = (args.get("pattern", "") or "").strip()
            query = (args.get("query", "") or "").strip()
            use_regex = bool(args.get("use_regex", False))
            case_sensitive = bool(args.get("case_sensitive", False))
            if query and not pattern:
                toks = [w for w in re.findall(r"\w+", query.lower(), re.UNICODE) if len(w) >= 2]
                if not toks:
                    return {"count": 0, "total": 0}
                hay = segment if case_sensitive else segment.lower()
                hits = sum(1 for t in set(toks[:12]) if t in hay)
                # Return how many distinct query tokens appear (proxy for usefulness)
                return {"count": hits, "total": len(set(toks[:12]))}

            if not pattern:
                return {"count": 0, "total": 0}
            hay = segment if case_sensitive else segment.lower()
            needle = pattern if case_sensitive else pattern.lower()
            if use_regex:
                flags = 0 if case_sensitive else re.IGNORECASE
                try:
                    rx = re.compile(pattern, flags)
                except re.error as ex:
                    return {"error": f"invalid regex: {ex}", "count": -1}
                return {"count": sum(1 for _ in rx.finditer(segment)), "total": len(segment)}
            return {"count": hay.count(needle), "total": len(segment)}

        # table_lookup: count lines containing the row string / regex.
        row = (args.get("row", "") or args.get("pattern", "") or "").strip()
        use_regex = bool(args.get("use_regex", False))
        case_sensitive = bool(args.get("case_sensitive", False))
        if not row:
            return {"count": 0, "total": 0}
        lines = segment.splitlines()
        if use_regex:
            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                rx = re.compile(row, flags)
            except re.error as ex:
                return {"error": f"invalid regex: {ex}", "count": -1}
            return {"count": sum(1 for ln in lines if rx.search(ln)), "total": len(lines)}
        needle = row if case_sensitive else row.lower()
        return {"count": sum(1 for ln in lines if needle in (ln if case_sensitive else ln.lower())), "total": len(lines)}

    return {"note": "non-deterministic, must execute to see result"}


def _extract_keywords(pattern: str) -> list[str]:
    """Extract tokens (4+ chars) from a regex pattern."""
    return list(dict.fromkeys(re.findall(r"[^\W_]{4,}", pattern, re.UNICODE)))


def _generate_regex_variants(pattern: str) -> list[str]:
    """Generate simpler regex variants from a complex pattern.

    Weak models write overly complex regexes that match nothing.
    This decomposes them into progressively simpler variants
    so the LLM can select one that matches a useful number of docs.
    """
    keywords = _extract_keywords(pattern)
    if not keywords:
        return []

    variants: list[str] = []
    seen: set[str] = {pattern.lower()}

    # Single keywords
    for kw in keywords[:4]:
        v = kw
        if v.lower() not in seen:
            variants.append(v)
            seen.add(v.lower())

    # Adjacent keyword pairs joined by .*
    for i in range(len(keywords) - 1):
        v = f"{keywords[i]}.*{keywords[i+1]}"
        if v.lower() not in seen:
            variants.append(v)
            seen.add(v.lower())
        if len(variants) >= 6:
            break

    # OR of all keywords
    if len(keywords) >= 2:
        v = "|".join(keywords[:4])
        if v.lower() not in seen:
            variants.append(v)

    return variants


def _dedup_by_count(
    alts: list[tuple[str, dict, dict]],
) -> list[tuple[str, dict, dict]]:
    """Remove alternatives that have the same doc count as another."""
    seen_counts: set[int] = set()
    deduped = []
    for item in alts:
        c = item[2].get("count", -1)
        if c not in seen_counts:
            seen_counts.add(c)
            deduped.append(item)
    return deduped


def generate_alternatives(
    ctx: "ToolContext", tool_name: str, args: dict, max_alts: int = 5,
) -> list[tuple[str, dict, dict]]:
    """Generate alternative actions with previewed outcomes.

    Returns list of (tool_name, args, preview_result) tuples.
    Does NOT include the proposed action — caller handles that.
    """
    alts: list[tuple[str, dict, dict]] = []
    named_sets = [
        n for n in ctx.sets._sets
        if n != "all" and not n.endswith("_neg")
    ]

    if tool_name == "apply_filter" and args.get("filter_type") == "regex":
        pattern = args.get("pattern", "")
        target_set = args.get("target_set", "all")
        keywords = _extract_keywords(pattern)

        proposed_count = preview_action(ctx, tool_name, args).get("count", -1)
        try:
            target_total = len(ctx.sets.resolve_ids(target_set))
        except KeyError:
            target_total = -1

        # Regex variants (simpler patterns)
        for variant in _generate_regex_variants(pattern):
            alt_args = {**args, "pattern": variant}
            preview = preview_action(ctx, "apply_filter", alt_args)
            c = preview.get("count", 0)
            if target_total > 0 and 0 < c < target_total and c != proposed_count:
                alts.append(("apply_filter", alt_args, preview))

        # Same pattern on different target sets
        for sname in named_sets[-3:]:
            if sname != target_set:
                alt_args = {**args, "target_set": sname}
                preview = preview_action(ctx, "apply_filter", alt_args)
                if preview.get("count", 0) > 0:
                    alts.append(("apply_filter", alt_args, preview))

        # Filename search with keywords
        for kw in keywords[:2]:
            alt_args = {"pattern": kw}
            preview = preview_action(ctx, "search_filenames", alt_args)
            c = preview.get("count", 0)
            if 0 < c < ctx.doc_store.doc_count():
                alts.append(("search_filenames", alt_args, preview))

        alts = _dedup_by_count(alts)

    elif tool_name == "search_filenames":
        pattern = args.get("pattern", "")
        keywords = _extract_keywords(pattern)

        proposed_count = preview_action(ctx, tool_name, args).get("count", -1)

        # Simplified filename searches (critical when regex is too complex)
        for kw in keywords[:3]:
            if kw.lower() != pattern.strip().lower():
                alt_args = {"pattern": kw}
                preview = preview_action(ctx, "search_filenames", alt_args)
                c = preview.get("count", 0)
                if 0 < c < ctx.doc_store.doc_count() and c != proposed_count:
                    alts.append(("search_filenames", alt_args, preview))

        # Keyword pairs
        for i in range(min(len(keywords) - 1, 2)):
            combo = f"{keywords[i]}.*{keywords[i+1]}"
            alt_args = {"pattern": combo, "use_regex": True}
            preview = preview_action(ctx, "search_filenames", alt_args)
            c = preview.get("count", 0)
            if 0 < c < ctx.doc_store.doc_count() and c != proposed_count:
                alts.append(("search_filenames", alt_args, preview))

        # Content-based regex as fallback
        for kw in keywords[:1]:
            alt_args = {
                "filter_type": "regex",
                "pattern": kw,
                "target_set": "all",
            }
            preview = preview_action(ctx, "apply_filter", alt_args)
            if 0 < preview.get("count", 0) < ctx.doc_store.doc_count():
                alts.append(("apply_filter", alt_args, preview))

        alts = _dedup_by_count(alts)

    elif tool_name == "cosine_search":
        query = args.get("query", "")
        keywords = _extract_keywords(query)

        # Alt: regex filter instead of cosine
        if keywords:
            pat = "|".join(keywords[:3])
            alt_args = {
                "filter_type": "regex",
                "pattern": pat,
                "target_set": "all",
            }
            preview = preview_action(ctx, "apply_filter", alt_args)
            alts.append(("apply_filter", alt_args, preview))

        # Alt: filename search
        for kw in keywords[:2]:
            alt_args = {"pattern": kw}
            preview = preview_action(ctx, "search_filenames", alt_args)
            if 0 < preview.get("count", 0) < ctx.doc_store.doc_count():
                alts.append(("search_filenames", alt_args, preview))

    elif tool_name == "set_operation":
        set_a = args.get("set_a", "")
        set_b = args.get("set_b", "")

        # Alt: all three operations previewed
        for op in ["intersect", "union", "difference"]:
            if op != args.get("operation", ""):
                alt_args = {**args, "operation": op}
                preview = preview_action(ctx, "set_operation", alt_args)
                if preview.get("count", 0) >= 0:
                    alts.append(("set_operation", alt_args, preview))

    elif tool_name == "find_in_doc":
        pattern = (args.get("pattern", "") or "").strip()
        query = (args.get("query", "") or "").strip()
        use_regex = bool(args.get("use_regex", False))

        proposed_count = preview_action(ctx, tool_name, args).get("count", -1)

        # If pattern-based search fails, try ranking mode (query=...).
        if pattern and not query:
            alt_args = {**args}
            alt_args.pop("pattern", None)
            alt_args["query"] = pattern
            preview = preview_action(ctx, "find_in_doc", alt_args)
            c = preview.get("count", 0)
            if c > 0 and c != proposed_count:
                alts.append(("find_in_doc", alt_args, preview))

        # Simpler substring variants
        for kw in _extract_keywords(pattern or query)[:4]:
            alt_args = {**args, "pattern": kw}
            alt_args.pop("query", None)
            preview = preview_action(ctx, "find_in_doc", alt_args)
            c = preview.get("count", 0)
            if c > 0 and c != proposed_count:
                alts.append(("find_in_doc", alt_args, preview))

        # Try regex mode if substring fails
        if pattern and not use_regex:
            alt_args = {**args, "use_regex": True}
            preview = preview_action(ctx, "find_in_doc", alt_args)
            c = preview.get("count", 0)
            if c > 0 and c != proposed_count:
                alts.append(("find_in_doc", alt_args, preview))

        alts = _dedup_by_count(alts)

    elif tool_name == "table_lookup":
        row = (args.get("row", "") or args.get("pattern", "") or "").strip()
        use_regex = bool(args.get("use_regex", False))
        proposed_count = preview_action(ctx, tool_name, args).get("count", -1)

        for kw in _extract_keywords(row)[:4]:
            alt_args = {**args, "row": kw}
            preview = preview_action(ctx, "table_lookup", alt_args)
            c = preview.get("count", 0)
            if c > 0 and c != proposed_count:
                alts.append(("table_lookup", alt_args, preview))

        if row and not use_regex:
            alt_args = {**args, "use_regex": True}
            preview = preview_action(ctx, "table_lookup", alt_args)
            c = preview.get("count", 0)
            if c > 0 and c != proposed_count:
                alts.append(("table_lookup", alt_args, preview))

        alts = _dedup_by_count(alts)

    return alts[:max_alts]


def format_branch_prompt(
    proposed_tool: str,
    proposed_args: dict,
    proposed_preview: dict,
    alternatives: list[tuple[str, dict, dict]],
) -> str:
    """Format branch options for the LLM to select from."""
    lines = [
        "═══ BRANCH AND SELECT ═══",
        "Before executing, here are previewed outcomes of your action and alternatives.",
        "Select the best option by number, or describe a different action.",
        "",
    ]

    def _fmt_preview(tool: str, args: dict, preview: dict) -> str:
        if preview.get("error"):
            return f"  → Error: {preview['error']}"

        count = preview.get("count", "?")
        total = preview.get("total", "")
        total_str = f"/{total}" if total else ""
        samples = preview.get("samples", [])
        note = preview.get("note", "")

        parts = []
        if count != "?":
            parts.append(f"{count}{total_str} docs")
        if samples:
            parts.append(f"e.g. {', '.join(samples[:2])}")
        if note:
            parts.append(note)
        return "  → " + "; ".join(parts) if parts else "  → (no preview)"

    def _fmt_action(tool: str, args: dict) -> str:
        if tool == "apply_filter":
            pat = args.get("pattern", args.get("filter_id", "?"))
            tgt = args.get("target_set", "all")
            return f'apply_filter(pattern="{pat}", target_set="{tgt}")'
        if tool == "search_filenames":
            return f'search_filenames("{args.get("pattern", "?")}")'
        if tool == "set_operation":
            return (f'set_operation({args.get("operation","?")}, '
                    f'{args.get("set_a","?")}, {args.get("set_b","?")})')
        if tool == "cosine_search":
            return f'cosine_search("{args.get("query", "?")[:50]}")'
        if tool == "find_in_doc":
            if args.get("query") and not args.get("pattern"):
                return f'find_in_doc(query="{(args.get("query") or "")[:40]}")'
            return f'find_in_doc(pattern="{(args.get("pattern") or "")[:40]}")'
        if tool == "table_lookup":
            row = (args.get("row") or args.get("pattern") or "")[:40]
            return f'table_lookup(row="{row}")'
        return f'{tool}({args})'

    lines.append(f"[1] YOUR PROPOSAL: {_fmt_action(proposed_tool, proposed_args)}")
    lines.append(_fmt_preview(proposed_tool, proposed_args, proposed_preview))

    for i, (tool, args, preview) in enumerate(alternatives, 2):
        lines.append(f"[{i}] {_fmt_action(tool, args)}")
        lines.append(_fmt_preview(tool, args, preview))

    lines.append("")
    lines.append(
        'Respond with JSON: {"selection": <number>, "reason": "brief reason"}. '
        "Or propose a completely different action."
    )

    return "\n".join(lines)
