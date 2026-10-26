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

"""Pre-computed extraction index from a pattern library.

Runs every validated pattern against every document in the corpus and
stores the results.  At inference time the agent gets:

1. Pre-registered named sets (documents matching each pattern)
2. A corpus profile (what structures exist and how prevalent they are)
3. A pattern_search tool (parameterized queries on the index)
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .patterns import PatternLibrary, PatternTemplate


_SCAN_CHAR_LIMIT = 12_000
_PER_DOC_TIME_BUDGET = 1.5  # seconds
_PER_PATTERN_TIME_BUDGET = 30  # seconds


def _scannable_text(text: str, limit: int = _SCAN_CHAR_LIMIT) -> str:
    """Return the head and tail of *text* for pattern scanning.

    Structured metadata (footers, headers, participant lists) often
    appears at the very end of a document, well past a simple prefix
    truncation.  By concatenating the first *limit* and last *limit*
    characters (with a separator to prevent false cross-boundary
    matches) we cover both ends without scanning the full body.

    For documents shorter than 2 * limit, returns the full text.
    """
    if len(text) <= 2 * limit:
        return text
    return text[:limit] + "\n\n" + text[-limit:]


def _value_in_text(value: str | list, text: str) -> bool:
    """Check if a parameter value appears in text (regex or literal).

    Accepts list values (e.g. ["Smith", "John"]) — all must match.
    """
    if isinstance(value, list):
        return all(_value_in_text(v, text) for v in value)
    try:
        return bool(re.search(value, text, re.IGNORECASE))
    except re.error:
        return value.lower() in text.lower()



def _safe_search(compiled: re.Pattern, text: str) -> re.Match | None:
    """Run re.search on head+tail scannable window of text."""
    try:
        return compiled.search(_scannable_text(text))
    except (RecursionError, OverflowError):
        return None


@dataclass
class PatternIndex:
    """Index of all matches for a single pattern across the corpus."""
    pattern_name: str
    description: str
    parameters: list[str]
    regex_template: str
    scope_regex: str
    matching_doc_ids: list[str]
    total_docs: int
    unique_values: dict[str, list[str]]
    # Captured groups keyed by document. New indexes persist these so
    # pattern_search never has to rerun regex over a large active set.
    # None denotes a legacy index that predates capture caching.
    doc_extracts: dict[str, dict[str, str]] | None = None
    value_keywords: dict[str, list[str]] | None = None
    # Pre-computed gap analysis: for each value keyword, clusters of
    # docs where the keyword appears in the doc but NOT inside the
    # captured region.  Computed at build time so inference is instant.
    # Structure: {keyword: {"confirmed": N, "clusters": [{"template": str, "count": N, "doc_ids": [...]}]}}
    keyword_gaps: dict[str, dict] | None = None


class ExtractionIndex:
    """Pre-computed index of pattern matches across a corpus.

    Built once during preprocessing. Serializable to JSON for storage
    in S3/GCS alongside the corpus index.

    Also stores filename-derived document categories.
    """

    def __init__(self) -> None:
        self._patterns: dict[str, PatternIndex] = {}
        self._doc_patterns: dict[str, list[str]] = defaultdict(list)
        self._filename_categories: dict[str, list[str]] = {}

    @property
    def pattern_names(self) -> list[str]:
        return list(self._patterns.keys())

    def get_pattern(self, name: str) -> PatternIndex | None:
        return self._patterns.get(name)

    def matching_docs(self, pattern_name: str) -> list[str]:
        pi = self._patterns.get(pattern_name)
        return pi.matching_doc_ids if pi else []

    def patterns_for_doc(self, doc_id: str) -> list[str]:
        return self._doc_patterns.get(doc_id, [])

    @property
    def filename_categories(self) -> dict[str, list[str]]:
        return self._filename_categories

    def search(
        self,
        pattern_name: str,
        params: dict[str, str],
        doc_store: Any,
        target_doc_ids: list[str] | None = None,
    ) -> list[dict]:
        """Run a parameterized pattern search at inference time.

        The regex template captures named groups.  When the caller
        provides params, we match the structural pattern and then
        POST-FILTER: each requested param value must appear
        (case-insensitive) in the corresponding captured group.

        Returns only confirmed matches where the param value is inside
        the captured region.  Gap analysis (keyword present in doc but
        not captured) is pre-computed at build time in keyword_gaps.
        """
        pi = self._patterns.get(pattern_name)
        if not pi:
            return []

        matching = pi.matching_doc_ids
        if target_doc_ids is None:
            candidates = matching
        else:
            target = set(target_doc_ids)
            candidates = [doc_id for doc_id in matching if doc_id in target]

        compiled = None
        if pi.doc_extracts is None:
            try:
                compiled = re.compile(
                    pi.regex_template, re.IGNORECASE | re.DOTALL
                )
            except re.error:
                return []

        results = []
        for doc_id in candidates:
            if pi.doc_extracts is not None:
                groups = pi.doc_extracts.get(doc_id)
                if groups is None:
                    continue
            else:
                try:
                    text = doc_store.get_text(doc_id)
                except KeyError:
                    continue
                m = _safe_search(compiled, text)
                if not m:
                    continue
                groups = {
                    k: (v[:2000] if v else "")
                    for k, v in m.groupdict().items()
                }
            if params:
                skip = False
                for param_key, param_val in params.items():
                    extracted = groups.get(param_key, "") or ""
                    if not _value_in_text(param_val, extracted):
                        skip = True
                        break
                if skip:
                    continue
            extract = {
                k: (v[:2000] if v else "") for k, v in groups.items()
            }
            meta = doc_store.get_meta(doc_id)
            results.append({
                "doc_id": doc_id,
                "filename": meta.filename,
                "extract": extract,
            })
        return results

    def corpus_profile(self, total_docs: int) -> str:
        """Generate a human-readable corpus structure summary."""
        lines = []
        for name in sorted(self._patterns):
            pi = self._patterns[name]
            n = len(pi.matching_doc_ids)
            pct = n / total_docs * 100 if total_docs else 0
            line = f"• {name} ({n}/{total_docs} docs, {pct:.0f}%): {pi.description}"
            if pi.unique_values:
                for param, vals in pi.unique_values.items():
                    shown = vals[:8]
                    suffix = f" (+{len(vals)-8} more)" if len(vals) > 8 else ""
                    line += f"\n    {param}: {', '.join(shown)}{suffix}"
            lines.append(line)
        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {"_patterns": {}, "_filename_categories": {}}
        for name, pi in self._patterns.items():
            d: dict[str, Any] = {
                "pattern_name": pi.pattern_name,
                "description": pi.description,
                "parameters": pi.parameters,
                "regex_template": pi.regex_template,
                "scope_regex": pi.scope_regex,
                "matching_doc_ids": pi.matching_doc_ids,
                "total_docs": pi.total_docs,
                "unique_values": pi.unique_values,
            }
            if pi.value_keywords:
                d["value_keywords"] = pi.value_keywords
            if pi.keyword_gaps:
                d["keyword_gaps"] = pi.keyword_gaps
            if pi.doc_extracts is not None:
                d["doc_extracts"] = pi.doc_extracts
            data["_patterns"][name] = d
        data["_filename_categories"] = self._filename_categories
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ExtractionIndex":
        path = Path(path)
        data = json.loads(path.read_text())
        idx = cls()

        # Support both old format (flat dict of patterns) and new
        patterns = data.get("_patterns", data)
        if "_patterns" in data:
            patterns = data["_patterns"]
            idx._filename_categories = data.get("_filename_categories", {})

        for name, d in patterns.items():
            if name.startswith("_"):
                continue
            if "value_keywords" not in d:
                d["value_keywords"] = None
            if "keyword_gaps" not in d:
                d["keyword_gaps"] = None
            if "doc_extracts" not in d:
                d["doc_extracts"] = None
            d.pop("near_miss_doc_ids", None)
            d.pop("after_match_types", None)
            pi = PatternIndex(**d)
            if pi.value_keywords is None and pi.unique_values:
                kws: dict[str, set[str]] = defaultdict(set)
                for param, vals in pi.unique_values.items():
                    for v in vals:
                        for tok in re.findall(r"[^\W\d_]{3,}", v, re.UNICODE):
                            kws[param].add(tok.lower())
                pi.value_keywords = {k: sorted(v) for k, v in kws.items()}
            idx._patterns[name] = pi
            for doc_id in pi.matching_doc_ids:
                idx._doc_patterns[doc_id].append(name)
        return idx

    def __len__(self) -> int:
        return len(self._patterns)

    def __repr__(self) -> str:
        return f"ExtractionIndex({len(self._patterns)} patterns)"


def _discover_filename_categories(
    doc_store: Any, verbose: bool = True
) -> dict[str, list[str]]:
    """Automatically discover document categories from filenames.

    Extracts the longest common structural segments from filenames.
    For example, filenames like "2024-01-09_Category_A_001.pdf" and
    "2024-02-05_Category_B_002.pdf" would yield categories
    {"Category_A": [...], "Category_B": [...]}.
    """
    all_ids = doc_store.all_doc_ids()
    if not all_ids:
        return {}

    filenames = {}
    for did in all_ids:
        try:
            meta = doc_store.get_meta(did)
            filenames[did] = meta.filename
        except (KeyError, AttributeError):
            continue

    if not filenames:
        return {}

    # Extract potential category tokens from filenames by finding
    # segments that recur across multiple files.  We split on common
    # delimiters, skip date-like and numeric-only tokens, and count
    # which token sequences appear across many files.
    from collections import Counter
    token_counter: Counter = Counter()
    doc_tokens: dict[str, list[str]] = {}

    for did, fn in filenames.items():
        base = fn.rsplit(".", 1)[0] if "." in fn else fn
        # Split on common delimiters
        parts = re.split(r"[_\-/\\. ]+", base)
        # Keep tokens that aren't purely numeric or date-like
        tokens = [
            p for p in parts
            if p and not re.match(r"^\d+$", p)
            and not re.match(r"^\d{4}$", p)
        ]
        doc_tokens[did] = tokens
        for tok in set(tokens):
            token_counter[tok] += 1

    n_docs = len(filenames)
    # Find tokens that appear in multiple docs but not all —
    # these are category indicators
    category_tokens = {
        tok: count for tok, count in token_counter.items()
        if 3 <= count < n_docs * 0.95
    }

    if not category_tokens:
        return {}

    # Build ngrams (2-3 tokens) that form category names
    ngram_counter: Counter = Counter()
    doc_ngrams: dict[str, list[str]] = {}

    for did, tokens in doc_tokens.items():
        ngrams = []
        for n in (3, 2, 1):
            for i in range(len(tokens) - n + 1):
                ngram = "_".join(tokens[i:i+n])
                if any(tokens[j] in category_tokens
                       for j in range(i, i+n)):
                    ngrams.append(ngram)
                    ngram_counter[ngram] += 1
        doc_ngrams[did] = ngrams

    # Pick the best ngram per doc — the longest one that recurs
    # at least 3 times across the corpus
    viable = {ng for ng, c in ngram_counter.items() if c >= 3}

    categories: dict[str, list[str]] = {}
    for did, ngrams in doc_ngrams.items():
        best = ""
        for ng in ngrams:
            if ng in viable and len(ng) > len(best):
                best = ng
        if best:
            categories.setdefault(best, []).append(did)

    # Filter out categories that are too small or too similar
    final = {
        cat: ids for cat, ids in categories.items()
        if len(ids) >= 3
    }

    if verbose and final:
        print(f"  Filename categories: {len(final)} types discovered")
        for cat in sorted(final, key=lambda c: -len(final[c]))[:10]:
            print(f"    {cat}: {len(final[cat])} docs")

    return final



def build_extraction_index(
    library: PatternLibrary,
    doc_store: Any,
    min_score: float = 0.5,
    verbose: bool = True,
) -> ExtractionIndex:
    """Build an extraction index from validated patterns.

    For each pattern passing the score threshold, instantiate the
    wildcard regex and scan every document.  Skips patterns whose
    regex is slow (per-pattern time budget).
    """
    index = ExtractionIndex()
    all_ids = doc_store.all_doc_ids()
    total = len(all_ids)

    templates = [
        t for t in library.all_templates()
        if t.validation_score >= min_score
    ]
    if verbose:
        print(f"Building extraction index: {len(templates)} patterns, "
              f"{total} docs")

    packed_stream = hasattr(doc_store, "iter_texts")

    for ti, template in enumerate(templates):
        wildcard = template.instantiate()
        try:
            compiled = re.compile(wildcard, re.IGNORECASE | re.DOTALL)
        except re.error:
            if verbose:
                print(f"  [{ti+1}/{len(templates)}] {template.name}: "
                      f"SKIP (bad regex)")
            continue

        matching_ids: list[str] = []
        doc_extracts: dict[str, dict[str, str]] = {}
        unique_vals: dict[str, set[str]] = defaultdict(set)
        val_kws: dict[str, set[str]] = defaultdict(set)
        t_start = time.time()
        skipped = False

        if packed_stream:
            text_rows = doc_store.iter_texts(all_ids)
        else:
            def _rows():
                for current_id in all_ids:
                    try:
                        yield current_id, doc_store.get_text(current_id)
                    except KeyError:
                        continue
            text_rows = _rows()

        for doc_id, text in text_rows:

            m = _safe_search(compiled, text)
            if m:
                matching_ids.append(doc_id)
                doc_extracts[doc_id] = {
                    k: (v.strip()[:2000] if v else "")
                    for k, v in m.groupdict().items()
                }
                for k, v in m.groupdict().items():
                    if v and v.strip():
                        unique_vals[k].add(v.strip()[:80])
                        for tok in re.findall(r"[^\W\d_]{3,}", v, re.UNICODE):
                            val_kws[k].add(tok.lower())

            elapsed = time.time() - t_start
            time_budget = 1800 if packed_stream else _PER_PATTERN_TIME_BUDGET
            if elapsed > time_budget:
                skipped = True
                break

        if skipped:
            if verbose:
                print(f"  [{ti+1}/{len(templates)}] {template.name}: "
                      f"SKIP (slow, {len(matching_ids)} partial)")
            continue

        sorted_vals = {k: sorted(v)[:30] for k, v in unique_vals.items()}
        sorted_kws = {k: sorted(v) for k, v in val_kws.items()}

        # ── Second pass: pre-compute keyword gap analysis ──────────
        # For each value keyword, identify docs where the keyword
        # exists in the document but NOT in any captured group.
        # Extract the keyword's context line and cluster by template.
        all_kws: set[str] = set()
        for kw_set in val_kws.values():
            all_kws.update(kw_set)

        keyword_gaps: dict[str, dict] = {}
        # Gap clustering is a pattern-development diagnostic.  Re-reading
        # millions of matches is not required by inference and would add a
        # second corpus pass per pattern, so perform it only on ordinary-size
        # stores.
        if all_kws and matching_ids and total < 100_000:
            kw_confirmed: dict[str, int] = defaultdict(int)
            kw_gap_lines: dict[str, list[tuple[str, str]]] = defaultdict(list)

            for doc_id in matching_ids:
                try:
                    text = doc_store.get_text(doc_id)
                except KeyError:
                    continue
                text_scan = _scannable_text(text)
                text_lower = text_scan.lower()
                m = _safe_search(compiled, text)
                if not m:
                    continue
                groups = m.groupdict()
                captured_lower = " ".join(
                    (v or "").lower() for v in groups.values()
                )

                for kw in all_kws:
                    if kw not in text_lower:
                        continue
                    if kw in captured_lower:
                        kw_confirmed[kw] += 1
                    else:
                        idx = text_lower.find(kw)
                        ls = text_scan.rfind("\n", 0, idx) + 1
                        le = text_scan.find("\n", idx)
                        if le == -1:
                            le = len(text_scan)
                        ctx = text_scan[ls:le].strip()[:100]
                        kw_gap_lines[kw].append((doc_id, ctx))

            for kw, entries in kw_gap_lines.items():
                clusters: dict[str, list[str]] = defaultdict(list)
                for doc_id, line in entries:
                    tmpl = line.lower().replace(kw, "{}")
                    clusters[tmpl].append(doc_id)
                keyword_gaps[kw] = {
                    "confirmed": kw_confirmed.get(kw, 0),
                    "clusters": sorted(
                        [{"template": t, "count": len(ids), "doc_ids": ids}
                         for t, ids in clusters.items()],
                        key=lambda c: -c["count"],
                    ),
                }

        pi = PatternIndex(
            pattern_name=template.name,
            description=template.description,
            parameters=template.parameters,
            regex_template=template.regex_template,
            scope_regex=template.scope_regex,
            matching_doc_ids=matching_ids,
            total_docs=total,
            unique_values=sorted_vals,
            doc_extracts=doc_extracts,
            value_keywords=sorted_kws,
            keyword_gaps=keyword_gaps if keyword_gaps else None,
        )
        index._patterns[template.name] = pi
        for doc_id in matching_ids:
            index._doc_patterns[doc_id].append(template.name)

        n_gaps = sum(
            sum(c["count"] for c in g["clusters"])
            for g in keyword_gaps.values()
        ) if keyword_gaps else 0
        if verbose:
            pct = len(matching_ids) / total * 100 if total else 0
            print(f"  [{ti+1}/{len(templates)}] {template.name}: "
                  f"{len(matching_ids)} docs ({pct:.0f}%), "
                  f"{len(sorted_vals)} extracted fields, "
                  f"{len(keyword_gaps)} keywords with gaps ({n_gaps} gap docs)")

    fn_cats = (
        _discover_filename_categories(doc_store, verbose=verbose)
        if total < 100_000 else {}
    )
    index._filename_categories = fn_cats

    if verbose:
        print(f"Extraction index built: {len(index)} patterns indexed, "
              f"{len(fn_cats)} filename categories")
    return index
