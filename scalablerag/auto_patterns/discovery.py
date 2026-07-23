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

"""Pattern discovery with corpus-scale validation.

Preprocessing step: run field scanning and LLM validation before eval so
inference can use `pattern_search` on pre-built regex templates.

Pipeline:
1. **Explore**: Sample documents; read content in chunks sized to the LLM context.
2. **Hypothesize**: LLM proposes parameterized regex templates.
3. **Safety gate**: Compilation, backtracking limits, bounded runtime.
4. **Corpus-scale scan**: Run each regex against all documents.
5. **Deterministic recall**: Extract literal keywords from the regex,
   search the corpus for them, compare with regex matches (false negatives).
6. **Precision check**: LLM verifies match contexts in batched windows.
7. **Refinement**: On low recall, show missed documents and re-prompt until
   precision and recall meet targets.
8. **Multiple rounds**: Up to 8 refinement rounds per pattern (wall-clock capped).
   Each round re-scans the full corpus.

Documents are chunked for small-context models; each chunk is scored independently.
"""

from __future__ import annotations

import json
import multiprocessing
import random
import re
import time
from typing import Any

from ..llm.base import BaseLLM
from ..storage.doc_store import DocStore
from .patterns import PatternLibrary, PatternTemplate


# -----------------------------------------------------------------------
# Regex safety
# -----------------------------------------------------------------------

_NESTED_QUANTIFIERS = re.compile(
    r"\([^)]*[+*][^)]*\)[+*?]"
    r"|"
    r"\([^)]*\([^)]*[+*][^)]*\)[^)]*\)[+*?]"
)

_MAX_REGEX_CHARS = 500
_SCAN_CHAR_LIMIT = 50_000
_SAFETY_TIME_BUDGET = 2.0


def _regex_safe(regex_str: str) -> bool:
    """Reject regexes that are too long, invalid, or likely to backtrack."""
    if len(regex_str) > _MAX_REGEX_CHARS:
        return False
    try:
        re.compile(regex_str, re.IGNORECASE | re.DOTALL)
    except re.error:
        return False
    if _NESTED_QUANTIFIERS.search(regex_str):
        return False
    return True


def _subprocess_search(pattern_str: str, flags: int, text: str) -> tuple:
    """Target for multiprocessing — runs re.search and returns match info.

    Must be a module-level function for pickling.
    """
    try:
        m = re.compile(pattern_str, flags).search(text)
        if m:
            return (True, m.start(), m.end(), m.groupdict())
        return (False, 0, 0, {})
    except (RecursionError, OverflowError, re.error):
        return (False, 0, 0, {})


class _MatchProxy:
    """Lightweight stand-in for re.Match returned from subprocess search."""

    __slots__ = ("_start", "_end", "_groupdict", "_text")

    def __init__(self, start: int, end: int, groupdict: dict, text: str):
        self._start = start
        self._end = end
        self._groupdict = groupdict
        self._text = text

    def start(self) -> int:
        return self._start

    def end(self) -> int:
        return self._end

    def groupdict(self) -> dict:
        return dict(self._groupdict)

    def group(self, n: int = 0) -> str:
        return self._text[self._start : self._end]


def _scannable_text(text: str, limit: int = _SCAN_CHAR_LIMIT) -> str:
    """Head + tail window for scanning both ends of long documents."""
    if len(text) <= 2 * limit:
        return text
    return text[:limit] + "\n\n" + text[-limit:]


def _timed_search(
    compiled: re.Pattern, text: str, budget: float = _SAFETY_TIME_BUDGET
) -> re.Match | _MatchProxy | None:
    """re.search with head+tail scannable window of text."""
    scannable = _scannable_text(text)
    try:
        return compiled.search(scannable)
    except (RecursionError, OverflowError):
        return None


def _safe_search_subprocess(
    compiled: re.Pattern, text: str, budget: float = _SAFETY_TIME_BUDGET
) -> _MatchProxy | None:
    """Run regex in a subprocess that can be killed on timeout.

    Use this for probe safety checks where we must guarantee termination.
    """
    scannable = _scannable_text(text)
    ctx = multiprocessing.get_context("fork")
    pool = ctx.Pool(1)
    try:
        result = pool.apply_async(
            _subprocess_search,
            (compiled.pattern, compiled.flags, scannable),
        )
        outcome = result.get(timeout=budget)
    except multiprocessing.TimeoutError:
        pool.terminate()
        return None
    except Exception:
        pool.terminate()
        return None
    finally:
        pool.close()
        pool.join()

    matched, start, end, groupdict = outcome
    if not matched:
        return None
    return _MatchProxy(start, end, groupdict, scannable)


def _probe_safety(compiled: re.Pattern, texts: list[str]) -> bool:
    """Run the regex on diverse texts in subprocesses.

    Uses subprocess execution so that catastrophic backtracking is
    detected by a hard kill rather than relying on the C-level regex
    engine to return.  Tests up to 10 texts for good coverage.
    """
    for text in texts[:10]:
        t0 = time.time()
        _safe_search_subprocess(compiled, text, budget=_SAFETY_TIME_BUDGET)
        if time.time() - t0 > _SAFETY_TIME_BUDGET:
            return False
    return True


# -----------------------------------------------------------------------
# Deterministic labeled-field detection (zero LLM cost)
# -----------------------------------------------------------------------

# Broadly matches anything that looks like "label: value" at the start
# of a line.  No language assumptions, no stopwords, no word-count
# limits.  Instead we filter by a data-driven criterion: value entropy.
# A real labeled field (like "Contact: X") has MANY distinct values
# across the corpus; a sentence fragment that happens to contain a
# colon (like "Chair Pyne adjourned the meeting at 8: 17 PM") has
# very few.
_LABEL_FIELD_RE = re.compile(
    r"^[ \t]*"
    r"([^\n:]{1,80}?)"       # candidate label: up to 80 chars, lazy
    r"\s*:\s*"                # colon + whitespace
    r"(\S[^\n]{0,300})",      # value: non-empty remainder of line
    re.MULTILINE | re.UNICODE,
)


def _discover_labeled_fields(doc_store: DocStore) -> list[dict]:
    """Scan ALL documents for recurring labeled fields (Key: Value).

    Returns candidate pattern dicts ready for the validation pipeline.
    This catches structured fields that LLM sampling might miss —
    especially fields that appear in a minority of documents.

    No hardcoded thresholds, stopwords, or language assumptions.

    Filtering criteria (all derived from the data):
      - **Recurrence**: the label appears in >=3 documents (the minimum
        needed to distinguish a recurring pattern from coincidence).
      - **Value variety**: the values following that label are not all
        identical.  If every occurrence has the same value, it is
        template boilerplate, not an extractable field.
      - **Structural deduplication**: labels that differ only by digits
        are collapsed (e.g. "opened at 7" and "opened at 8" become
        "opened at {N}") and only the highest-frequency variant per
        structural group is kept.  This eliminates near-duplicates
        caused by time/number formats without any language rules.
      - **At least 2 alphabetic characters** in the label (filters out
        bare numbers and punctuation).
    """
    from collections import Counter, defaultdict
    all_ids = doc_store.all_doc_ids()
    if not all_ids:
        return []

    label_doc_counts: Counter = Counter()
    label_values: dict[str, set[str]] = {}
    label_examples: dict[str, list[str]] = {}

    for did in all_ids:
        try:
            text = doc_store.get_text(did)
        except KeyError:
            continue
        seen_labels: set[str] = set()
        for m in _LABEL_FIELD_RE.finditer(_scannable_text(text)):
            raw_label = m.group(1).strip()
            raw_value = m.group(2).strip()
            if not raw_label or not raw_value:
                continue
            norm = re.sub(r"\s+", " ", raw_label).strip()
            key = norm.lower()
            if len(key) < 2:
                continue
            alpha_chars = sum(1 for c in key if c.isalpha())
            if alpha_chars < 2:
                continue

            if key not in seen_labels:
                seen_labels.add(key)
                label_doc_counts[key] += 1
                if key not in label_values:
                    label_values[key] = set()
                label_values[key].add(raw_value[:60].lower())
                if key not in label_examples:
                    label_examples[key] = []
                if len(label_examples[key]) < 3:
                    label_examples[key].append(m.group(0)[:200])

    # --- Structural deduplication ---
    # Labels that differ only in digits are the same structural pattern.
    # Group them; keep the most frequent variant per group.
    def _structural_key(label: str) -> str:
        return re.sub(r"\d+", "{N}", label)

    struct_groups: dict[str, list[str]] = defaultdict(list)
    for key in label_doc_counts:
        struct_groups[_structural_key(key)].append(key)

    # For each structural group, keep only the highest-count variant
    best_per_group: set[str] = set()
    for variants in struct_groups.values():
        best = max(variants, key=lambda k: label_doc_counts[k])
        best_per_group.add(best)

    n_docs = len(all_ids)

    candidates: list[dict] = []
    for key, count in label_doc_counts.most_common():
        # Recurrence: at least 3 documents
        if count < 3:
            break
        if key not in best_per_group:
            continue
        n_distinct = len(label_values.get(key, set()))
        # Value variety: at least 2 distinct values
        if n_distinct < 2:
            continue

        display_label = (
            label_examples[key][0].split(":")[0].strip()
            if label_examples.get(key) else key
        )
        safe_name = re.sub(r"[^a-z0-9]+", "_", key).strip("_")[:40]
        if not safe_name:
            continue

        candidates.append({
            "name": f"field_{safe_name}",
            "description": (
                f"Extracts the value of the '{display_label}' labeled field. "
                f"Found in {count}/{n_docs} documents ({count*100//n_docs}%). "
                f"{n_distinct} distinct values observed."
            ),
            "parameters": [],
            "regex_template": (
                rf"(?:^|[\n])[ \t]*{re.escape(display_label)}"
                r"\s*:\s*(?P<value>[^\n]{1,300})"
            ),
            "scope_regex": "",
            "output_type": "extract",
            "_doc_count": count,
            "_value_variety": n_distinct,
            "_examples": label_examples.get(key, []),
        })

    return candidates


# -----------------------------------------------------------------------
# LLM prompts for pattern discovery
# -----------------------------------------------------------------------

_DISCOVER_SYSTEM = """\
You are an expert at analyzing document corpora and discovering recurring \
structural patterns.  Derive all anchor text from the actual sample \
documents; do NOT add translations or alternate-language text.

Your goal: identify LARGE STRUCTURAL SECTIONS of documents as parameterized \
regex templates.  These patterns will be used by a search agent that passes \
parameter values at query time to find matching documents — so parameters \
MUST be useful as query filters.

WHAT MAKES A GOOD PATTERN:
- Captures a BROAD SECTION of recurring structure — not a single labeled \
  field but an entire MULTI-LINE REGION from one section heading to the \
  next: a participant/attendance block, a voting/decision section, an \
  action-items or discussion block, a metadata header.
- Has NAMED CAPTURE GROUPS (?P<name>...) that capture MEANINGFUL BLOCKS \
  of text that vary across documents (e.g. a multi-line list of names, \
  a block of items).  The agent checks whether a value appears INSIDE a \
  captured block, so groups must span wide enough regions.
- Parameters are the names of these capture groups.

CRITICAL: Do NOT create single-line "Label: Value" field extractors — \
those are already handled by a separate system.  Instead, capture the \
FULL MULTI-LINE BLOCK below a section heading.  The heading itself is \
just the anchor; the PAYLOAD is the entire block of text after it.

BAD PATTERNS (do NOT produce these):
- "Label: Value" field extractors (e.g. (?:Present\\s*:\\s*(?P<v>.+))).
- Patterns with zero named capture groups.
- Patterns named after specific people or text content.
- Overly broad patterns that match most of the document.

For each pattern, output:
- name: short snake_case identifier for the structural section
- description: what this pattern detects/extracts (1-2 sentences)
- parameters: list of NAMED CAPTURE GROUPS that serve as query filters
- regex_template: a Python regex with (?P<param_name>...) capture groups \
  AND optional {parameter_name} placeholders for runtime filtering
- scope_regex: leave empty if broadly applicable
- output_type: "boolean" | "extract" | "list"

REGEX RULES:
- Must work with Python re module (re.IGNORECASE | re.DOTALL).
- EVERY pattern MUST have at least one (?P<name>...) NAMED capture group. \
  Use (?:...) ONLY for sub-grouping inside a named group, NEVER as the \
  outermost repeating group.  WRONG: (?:line\\n)+  RIGHT: (?P<block>(?:line\\n)+)
- Use BOUNDED quantifiers: .{0,200} instead of .* or .+
- NEVER use nested quantifiers like (.+?)+ or (\\S+\\s*)+
- Prefer [^\\n]{0,200} over .{0,200} when matching within a single line.
- {parameter_name} placeholders get re.escape(value) at runtime.
- Account for whitespace, line-break, OCR, and formatting variations.
- Keep patterns simple and robust.  A simple pattern that works on 70% \
  of docs is better than a complex one that works on 90% but backtracks.

Respond with a JSON array of pattern objects.  Nothing else.\
"""

_DISCOVER_EXTRACTION = """\
You are an expert at analyzing document corpora and discovering information \
extraction patterns.  Derive all anchor text from the actual documents.

You already found some structural patterns in a previous round.  Now look \
for ADDITIONAL recurring MULTI-LINE SECTIONS you missed — especially \
sections that serve a DIFFERENT PURPOSE from the ones already found.

Think about what a search agent would need: sections listing people and \
their roles, sections recording outcomes or decisions, sections for items \
under discussion, sections with dates/deadlines/quantities.

Each pattern MUST capture a MULTI-LINE BLOCK (not a single labeled field). \
"Label: Value" extraction is handled by a separate system.

Same output format and regex rules as before.  Respond with a JSON array.\
"""

_PRECISION_CHECK = """\
You are validating a regex pattern against actual corpus matches.

PATTERN: {pattern_name}
DESCRIPTION: {description}

For each match below, judge STRICTLY:
- 1 = the matched text genuinely represents what the description says
- 0 = false positive (matched something unrelated, or extracted wrong value)

Output JSON: {{"ratings": [0 or 1, ...], "notes": "brief explanation"}}\
"""


_REFINE_PROMPT = """\
Improve this regex pattern based on validation feedback.

Pattern: {name}
Description: {description}
Parameters: {parameters}
Current regex: {regex_template}
Current scope regex: {scope_regex}

VALIDATION RESULTS:
- Matched {match_count}/{total_checked} docs ({match_rate})
- Precision: {precision} ({n_fp} false positives in {precision_n} checked)
- Recall: {recall} ({n_fn} false negatives in {recall_n} checked)

{priority_guidance}

{fp_section}
{fn_section}

{semantic_notes}

RULES: Keep the regex simple and bounded (.{{0,N}}, no nested quantifiers). \
Output ONLY a JSON object: {{"regex_template": "...", "scope_regex": "..."}}.\
"""


def _chars_per_token() -> float:
    return 3.5


# -----------------------------------------------------------------------
# Discovery engine
# -----------------------------------------------------------------------


class PatternDiscovery:
    """Discover and rigorously validate parameterized regex patterns."""

    def __init__(
        self,
        llm: BaseLLM,
        doc_store: DocStore,
        *,
        sample_size: int = 20,
        precision_sample: int = 25,
        recall_sample: int = 15,
        max_refinement_rounds: int = 8,
        target_precision: float = 0.90,
        target_recall: float = 0.85,
    ):
        self.llm = llm
        self.doc_store = doc_store
        self.sample_size = sample_size
        self.precision_sample = precision_sample
        self.recall_sample = recall_sample
        self.max_refinement_rounds = max_refinement_rounds
        self.target_precision = target_precision
        self.target_recall = target_recall

        ctx_tokens = self.llm.context_length
        self._ctx_chars = int(ctx_tokens * _chars_per_token())
        self._ctx_tokens = ctx_tokens

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def discover(self, verbose: bool = True) -> PatternLibrary:
        """Run the full discovery + validation pipeline."""
        t0 = time.time()
        library = PatternLibrary()
        all_ids = self.doc_store.all_doc_ids()

        if verbose:
            print(
                f"[discover] Corpus: {len(all_ids)} docs, "
                f"LLM ctx={self._ctx_tokens} tokens",
                flush=True,
            )

        # Phase 1: sample and read documents
        sample_ids = self._diverse_sample(self.sample_size)
        sample_docs = self._load_full_texts(sample_ids)

        if verbose:
            total_chars = sum(len(d["text"]) for d in sample_docs)
            print(
                f"[discover] Sampled {len(sample_docs)} docs "
                f"({total_chars:,} chars)",
                flush=True,
            )

        # Phase 1b: deterministic labeled-field scan (zero LLM cost)
        if verbose:
            print("[discover] Scanning corpus for labeled fields...", flush=True)
        field_candidates = _discover_labeled_fields(self.doc_store)
        if verbose:
            print(
                f"[discover] Found {len(field_candidates)} recurring labeled fields",
                flush=True,
            )
            for fc in field_candidates[:10]:
                print(
                    f"  {fc['name']}: {fc.get('_doc_count', '?')} docs "
                    f"— e.g. {(fc.get('_examples', [''])[0])[:80]}",
                    flush=True,
                )

        # Phase 2: LLM-driven pattern hypotheses
        if verbose:
            print("[discover] Generating LLM pattern hypotheses...", flush=True)

        raw = self._discover_ctx_aware(sample_docs, verbose)

        # --- Accept field-scanner candidates directly ---
        # Their regex is deterministic (generated from observed corpus
        # text, not LLM-hallucinated), so there are no capture-group
        # bugs to fix and no refinement needed.  Corpus statistics
        # (_doc_count, _value_variety) are already known.
        field_names: set[str] = set()
        for fc in field_candidates:
            template = self._raw_to_template(fc)
            if template is None:
                continue
            n_docs = len(all_ids)
            doc_count = fc.get("_doc_count", 0)
            variety = fc.get("_value_variety", 0)
            template.validation_score = 1.0
            template.validation_details = {
                "source": "field_scanner",
                "doc_count": doc_count,
                "corpus_size": n_docs,
                "value_variety": variety,
                "coverage_pct": round(doc_count * 100 / max(1, n_docs), 1),
            }
            library.add(template)
            field_names.add(template.name)

        if verbose:
            print(
                f"[discover] Accepted {len(field_names)} field-scanner "
                f"patterns (deterministic, no LLM validation needed)",
                flush=True,
            )

        # Deduplicate LLM candidates against already-accepted fields
        seen: set[str] = set(field_names)
        llm_unique: list[dict] = []
        for rp in raw:
            n = rp.get("name", "")
            if n and n not in seen:
                seen.add(n)
                llm_unique.append(rp)

        if verbose:
            print(
                f"[discover] {len(llm_unique)} LLM candidates to validate "
                f"(after dedup against field-scanner patterns)",
                flush=True,
            )

        # Preload probe texts for safety checks — use 20 diverse docs
        # to maximize the chance of catching pathological backtracking
        probe_ids = self._diverse_sample(min(20, len(all_ids)))
        probe_texts = [
            self.doc_store.get_text(did)
            for did in probe_ids
            if self.doc_store.get_text(did)
        ]

        # Phase 3-8: validate LLM-generated patterns only
        for rp in llm_unique:
            template = self._raw_to_template(rp)
            if template is None:
                continue

            final = self._validate_and_refine(
                template, all_ids, probe_texts, verbose
            )
            if final is not None:
                library.add(final)

        elapsed = time.time() - t0
        if verbose:
            n_pass = sum(
                1 for t in library.all_templates()
                if t.validation_score >= self.target_precision
            )
            print(
                f"\n[discover] Done in {elapsed:.1f}s: "
                f"{len(library)} patterns ({n_pass} well-validated)",
                flush=True,
            )
        return library

    # ------------------------------------------------------------------
    # Validation + refinement loop
    # ------------------------------------------------------------------

    _PER_PATTERN_WALL_CLOCK = 300  # seconds per pattern

    def _validate_and_refine(
        self,
        template: PatternTemplate,
        all_ids: list[str],
        probe_texts: list[str],
        verbose: bool,
    ) -> PatternTemplate | None:
        """Validate a pattern and refine iteratively until targets are met.

        Precision: LLM reads match contexts and judges correctness.
        Recall: LLM reads a sample of NON-matching docs and judges
        whether they contain the concept — concept-driven, not
        regex-derived, so it can find any format variant.
        Refinement: LLM sees actual false-negative text to fix the regex.
        """
        best_template: PatternTemplate | None = None
        best_score = -1.0
        pattern_start = time.time()

        for round_i in range(1 + self.max_refinement_rounds):
            if time.time() - pattern_start > self._PER_PATTERN_WALL_CLOCK:
                if verbose:
                    print(
                        f"  [{template.name}] wall clock exceeded, "
                        f"keeping best",
                        flush=True,
                    )
                break

            wildcard = template.instantiate()

            if not _regex_safe(wildcard):
                if round_i == 0:
                    if verbose:
                        print(
                            f"  [{template.name}] REJECT (regex unsafe)",
                            flush=True,
                        )
                    return None
                if verbose:
                    print(
                        f"  [{template.name}] r{round_i}: "
                        f"refined regex unsafe, keeping best",
                        flush=True,
                    )
                break

            try:
                compiled = re.compile(wildcard, re.IGNORECASE | re.DOTALL)
            except re.error:
                if round_i == 0:
                    if verbose:
                        print(
                            f"  [{template.name}] REJECT (compile error)",
                            flush=True,
                        )
                    return None
                break

            if not _probe_safety(compiled, probe_texts):
                if round_i == 0:
                    if verbose:
                        print(
                            f"  [{template.name}] REJECT (slow on probes)",
                            flush=True,
                        )
                    return None
                if verbose:
                    print(
                        f"  [{template.name}] r{round_i}: "
                        f"refined regex slow, keeping best",
                        flush=True,
                    )
                break

            scan = self._corpus_scan(compiled, all_ids)
            if scan["timed_out"]:
                if round_i == 0:
                    if verbose:
                        print(
                            f"  [{template.name}] REJECT (scan timeout)",
                            flush=True,
                        )
                    return None
                if verbose:
                    print(
                        f"  [{template.name}] r{round_i}: scan timeout, "
                        f"keeping best",
                        flush=True,
                    )
                break

            match_ids = scan["match_ids"]
            match_contexts = scan["match_contexts"]
            partial_match_ids = scan.get("partial_match_ids", [])

            if len(match_ids) == 0 and len(partial_match_ids) == 0:
                if round_i == 0:
                    if verbose:
                        print(
                            f"  [{template.name}] REJECT (0 matches)",
                            flush=True,
                        )
                    return None
                break

            if partial_match_ids and verbose:
                print(
                    f"  [{template.name}] r{round_i}: "
                    f"{len(partial_match_ids)} partial matches "
                    f"(regex fires but capture groups empty)",
                    flush=True,
                )

            # --- Precision via LLM verification ---
            prec, prec_details = self._check_precision(
                template, match_ids, match_contexts
            )

            # --- Recall via LLM concept check on non-matches ---
            non_match_ids = scan["non_match_ids"]
            false_neg_ids, recall_info = self._llm_recall_probe(
                template, match_ids, non_match_ids, verbose
            )
            # Partial matches are capture-group failures — treat as
            # false negatives for refinement so the LLM can fix the
            # regex to capture these format variants.
            all_fn_ids = false_neg_ids + partial_match_ids

            n_sampled = min(self.precision_sample, len(non_match_ids))
            total_fn_estimate = len(partial_match_ids)
            if n_sampled > 0:
                fn_rate = len(false_neg_ids) / n_sampled
                total_fn_estimate += int(fn_rate * len(non_match_ids))
            estimated_recall = (
                len(match_ids)
                / max(1, len(match_ids) + total_fn_estimate)
            )
            recall = round(estimated_recall, 3)

            score = min(prec, recall)
            template.validation_score = round(score, 3)

            details = {
                "match_count": len(match_ids),
                "partial_matches": len(partial_match_ids),
                "total_docs": len(all_ids),
                "match_rate": round(
                    len(match_ids) / max(1, len(all_ids)), 3),
                "precision": round(prec, 3),
                "recall": round(recall, 3),
                "recall_info": recall_info,
                "n_false_negatives": len(all_fn_ids),
                "precision_details": prec_details,
                "recall_details": {
                    "n_fn": len(all_fn_ids),
                    "fn_examples": [],
                },
            }
            template.validation_details = details

            if verbose:
                if prec >= self.target_precision and recall >= self.target_recall:
                    tag = "PASS"
                elif prec < self.target_precision and recall < self.target_recall:
                    tag = "refine (low prec+rec)"
                elif prec < self.target_precision:
                    tag = "refine (low prec)"
                else:
                    tag = f"refine (low recall: {len(false_neg_ids)} missed)"
                print(
                    f"  [{template.name}] r{round_i}: "
                    f"{len(match_ids)}/{len(all_ids)} matched "
                    f"({details['match_rate']:.0%}), "
                    f"precision={prec:.2f}, recall={recall:.2f} [{tag}]",
                    flush=True,
                )

            if score > best_score:
                best_score = score
                best_template = PatternTemplate(
                    name=template.name,
                    description=template.description,
                    parameters=list(template.parameters),
                    regex_template=template.regex_template,
                    scope_regex=template.scope_regex,
                    output_type=template.output_type,
                    validation_score=template.validation_score,
                    validation_details=dict(template.validation_details),
                )

            passed = (prec >= self.target_precision
                      and recall >= self.target_recall)
            if passed:
                break
            if round_i >= self.max_refinement_rounds:
                break

            match_rate = len(match_ids) / max(1, len(all_ids))
            if match_rate > 0.95 and prec < 0.10:
                if verbose:
                    print(
                        f"  [{template.name}] REJECT "
                        f"(too broad: {match_rate:.0%} matches, "
                        f"{prec:.0%} precision)",
                        flush=True,
                    )
                return best_template

            # --- Build refinement evidence from real documents ---
            # Include partial matches (capture-group failures) first
            # since they're the most informative for fixing the regex.
            fn_snippets = self._read_false_negatives(
                partial_match_ids + false_neg_ids, template, compiled
            )
            details["recall_details"]["fn_examples"] = fn_snippets

            refined = self._refine(template, details)
            if refined is None:
                break
            template.regex_template = self._fix_lazy_block_captures(
                refined.get("regex_template", template.regex_template)
            )
            if "scope_regex" in refined:
                template.scope_regex = refined["scope_regex"]

        return best_template

    # ------------------------------------------------------------------
    # LLM-based recall probe (concept-driven)
    # ------------------------------------------------------------------

    def _llm_recall_probe(
        self,
        template: PatternTemplate,
        match_ids: list[str],
        non_match_ids: list[str],
        verbose: bool,
    ) -> tuple[list[str], str]:
        """Concept-driven recall estimation via LLM.

        Sample non-matching docs and ask the LLM whether each contains
        the structural concept the pattern describes.  This is
        concept-driven (based on the pattern DESCRIPTION, not its regex
        keywords), so it can find ANY format variant — including ones
        that use completely different vocabulary.

        Returns (false_negative_ids, info_string).
        """
        n_sample = min(self.precision_sample, len(non_match_ids))
        if n_sample == 0:
            return [], "no non-matches to sample"

        sample = random.sample(non_match_ids, n_sample)

        rating_tokens = max(300, 40 * n_sample + 100)
        evidence_budget = self._doc_budget_for_request(rating_tokens)
        per_doc = max(600, evidence_budget // max(n_sample, 1))

        evidence_parts: list[str] = []
        sample_used: list[str] = []
        for i, doc_id in enumerate(sample):
            try:
                text = self.doc_store.get_text(doc_id)
                meta = self.doc_store.get_meta(doc_id)
            except KeyError:
                continue
            excerpt = text[:per_doc]
            evidence_parts.append(
                f"\nDoc {i+1} ({meta.filename}):\n{excerpt}\n"
            )
            sample_used.append(doc_id)

        if not evidence_parts:
            return [], "no evidence"

        prompt = (
            f"You are checking whether documents contain the structural "
            f"concept described by a pattern.\n\n"
            f"PATTERN: {template.name}\n"
            f"DESCRIPTION: {template.description}\n\n"
            f"For each document below, judge:\n"
            f"- 1 = this document DOES contain the described structure "
            f"(even if formatted differently from what a regex might expect)\n"
            f"- 0 = this document does NOT contain it\n\n"
            f'Output JSON: {{"ratings": [0 or 1, ...], '
            f'"notes": "brief explanation of any format variants found"}}'
        )

        all_ratings: list[int] = []
        all_notes: list[str] = []
        batch_size = max(3, n_sample)
        total_chars = sum(len(p) for p in evidence_parts)
        if total_chars > evidence_budget:
            batch_size = max(
                3, int(len(evidence_parts) * evidence_budget / total_chars)
            )

        for b_start in range(0, len(evidence_parts), batch_size):
            batch = evidence_parts[b_start : b_start + batch_size]
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": "".join(batch)},
            ]
            ratings, notes = self._ask_ratings(messages, len(batch))
            all_ratings.extend(ratings)
            if notes:
                all_notes.append(notes)

        fn_ids = [
            sample_used[i]
            for i, r in enumerate(all_ratings)
            if r == 1 and i < len(sample_used)
        ]

        n_checked = len(all_ratings)
        info = (
            f"LLM recall probe: {len(fn_ids)} false negatives "
            f"in {n_checked} non-matches sampled"
        )
        if all_notes:
            info += f" ({'; '.join(all_notes)})"
        if verbose:
            print(
                f"  [{template.name}] recall probe: "
                f"{len(fn_ids)}/{n_checked} non-matches contain concept",
                flush=True,
            )
        return fn_ids, info

    # ------------------------------------------------------------------
    # Read actual false-negative documents for refinement evidence
    # ------------------------------------------------------------------

    def _read_false_negatives(
        self,
        fn_ids: list[str],
        template: PatternTemplate,
        compiled: re.Pattern,
        max_examples: int = 10,
    ) -> list[dict]:
        """Read missed documents with context around keyword hits.

        For partial matches (regex fires but capture groups empty),
        includes the match span and group diagnosis.

        Uses larger context windows than inference (preprocessing only).
        """
        examples: list[dict] = []
        candidates = list(fn_ids)
        random.shuffle(candidates)

        wildcard = template.instantiate()
        for p in template.parameters:
            wildcard = wildcard.replace(f"{{{p}}}", "")
        literals = re.findall(r"[^\W\d_]{3,}", wildcard, re.UNICODE)
        literals.sort(key=len, reverse=True)

        for doc_id in candidates:
            if len(examples) >= max_examples:
                break
            try:
                text = self.doc_store.get_text(doc_id)
                meta = self.doc_store.get_meta(doc_id)
            except KeyError:
                continue

            # Check if this is a partial match (regex fires, groups empty)
            partial_diag = ""
            try:
                m = _safe_search_subprocess(compiled, text)
                if m:
                    groups = m.groupdict()
                    filled = any(v and v.strip() for v in groups.values())
                    if not filled:
                        match_span = text[m.start():m.end()][:500]
                        group_status = {
                            k: repr(v)[:60] for k, v in groups.items()
                        }
                        partial_diag = (
                            f"  PARTIAL MATCH DIAGNOSIS:\n"
                            f"  The regex matched this text span:\n"
                            f"    \"{match_span}\"\n"
                            f"  But named groups captured:\n"
                            f"    {group_status}\n"
                            f"  Fix: adjust the capture groups to "
                            f"actually capture from this text.\n"
                        )
            except Exception:
                pass

            regions: list[str] = []
            text_lower = text.lower()
            for lit in literals[:3]:
                lit_lower = lit.lower()
                search_start = 0
                while search_start < len(text_lower):
                    idx = text_lower.find(lit_lower, search_start)
                    if idx < 0:
                        break
                    r_start = max(0, idx - 400)
                    r_end = min(len(text), idx + 1200)
                    regions.append(text[r_start:r_end])
                    search_start = idx + len(lit_lower)
                if regions:
                    break

            if not regions:
                regions = [text[:1500]]

            if partial_diag:
                regions[0] = partial_diag + "\n" + regions[0]

            examples.append({
                "doc_id": doc_id,
                "filename": meta.filename,
                "regions": regions[:3],
                "n_total_occurrences": len(regions),
            })

        return examples

    # ------------------------------------------------------------------
    # Corpus scan — uses same approach as agent's apply_filter
    # ------------------------------------------------------------------

    def _corpus_scan(
        self, compiled: re.Pattern, all_ids: list[str]
    ) -> dict[str, Any]:
        """Scan the entire corpus; return matched/non-matched ID lists
        and match contexts (for precision checking).

        Also detects *partial matches* — docs where the regex fires but
        all named capture groups are empty.  These indicate format variants
        that the regex sees structurally but doesn't capture properly.

        Uses a subprocess pool so that catastrophic backtracking on any
        single document triggers a hard kill instead of hanging the
        process forever.
        """
        match_ids: list[str] = []
        non_match_ids: list[str] = []
        match_contexts: dict[str, str] = {}
        partial_match_ids: list[str] = []

        t_global = time.time()
        timed_out = False

        ctx = multiprocessing.get_context("fork")
        pool = ctx.Pool(1)
        pattern_str = compiled.pattern
        flags = compiled.flags

        try:
            for doc_id in all_ids:
                try:
                    text = self.doc_store.get_text(doc_id)
                except KeyError:
                    continue

                truncated = _scannable_text(text)
                t_doc = time.time()
                try:
                    result = pool.apply_async(
                        _subprocess_search,
                        (pattern_str, flags, truncated),
                    )
                    outcome = result.get(timeout=_SAFETY_TIME_BUDGET)
                except multiprocessing.TimeoutError:
                    pool.terminate()
                    timed_out = True
                    break
                except Exception:
                    pool.terminate()
                    timed_out = True
                    break

                doc_elapsed = time.time() - t_doc
                if doc_elapsed > _SAFETY_TIME_BUDGET:
                    timed_out = True
                    break

                matched, start_pos, end_pos, groupdict = outcome
                if matched:
                    filled = any(
                        v and v.strip() for v in groupdict.values()
                    )
                    if filled:
                        match_ids.append(doc_id)
                    else:
                        partial_match_ids.append(doc_id)
                    ctx_start = max(0, start_pos - 500)
                    ctx_end = min(len(text), end_pos + 1000)
                    match_contexts[doc_id] = text[ctx_start:ctx_end][:1500]
                else:
                    non_match_ids.append(doc_id)

                if time.time() - t_global > 60:
                    timed_out = True
                    break
        finally:
            pool.close()
            pool.join()

        return {
            "match_ids": match_ids,
            "non_match_ids": non_match_ids,
            "partial_match_ids": partial_match_ids,
            "match_contexts": match_contexts,
            "timed_out": timed_out,
        }

    # ------------------------------------------------------------------
    # Precision check: spot-check positive matches
    # ------------------------------------------------------------------

    def _check_precision(
        self,
        template: PatternTemplate,
        match_ids: list[str],
        match_contexts: dict[str, str],
    ) -> tuple[float, dict]:
        """Estimate precision by sampling matches and LLM-judging contexts."""
        n_check = min(self.precision_sample, len(match_ids))
        sample = random.sample(match_ids, n_check)

        rating_tokens = max(300, 40 * n_check + 100)
        evidence_budget = self._doc_budget_for_request(rating_tokens)
        per_doc = max(600, evidence_budget // max(n_check, 1))

        evidence_parts: list[str] = []
        sample_used: list[str] = []
        for i, doc_id in enumerate(sample):
            # Context window around the match
            ctx = match_contexts.get(doc_id, "")
            if len(ctx) < per_doc:
                try:
                    text = self.doc_store.get_text(doc_id)
                    if ctx:
                        pos = text.find(ctx[:50])
                        if pos >= 0:
                            start = max(0, pos - 300)
                            end = min(len(text), pos + per_doc)
                            ctx = text[start:end]
                        else:
                            ctx = text[:per_doc]
                    else:
                        ctx = text[:per_doc]
                except KeyError:
                    continue
            ctx = ctx[:per_doc]
            meta = self.doc_store.get_meta(doc_id)
            evidence_parts.append(
                f"\nMatch {i+1} (doc: {meta.filename}):\n{ctx}\n"
            )
            sample_used.append(doc_id)

        if not evidence_parts:
            return 1.0, {"n_checked": 0}

        # If evidence is too large for one call, split into batches
        all_ratings: list[int] = []
        all_notes: list[str] = []
        batch_size = max(3, n_check)
        total_chars = sum(len(p) for p in evidence_parts)
        if total_chars > evidence_budget:
            batch_size = max(
                3, int(len(evidence_parts) * evidence_budget / total_chars)
            )

        for b_start in range(0, len(evidence_parts), batch_size):
            batch = evidence_parts[b_start : b_start + batch_size]
            prompt = _PRECISION_CHECK.format(
                pattern_name=template.name,
                description=template.description,
            )
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": "".join(batch)},
            ]
            ratings, notes = self._ask_ratings(messages, len(batch))
            all_ratings.extend(ratings)
            if notes:
                all_notes.append(notes)

        if not all_ratings:
            return 0.8, {"n_checked": 0, "fallback": True}

        prec = sum(all_ratings) / len(all_ratings)
        fp_indices = [i for i, r in enumerate(all_ratings) if r == 0]
        fp_examples = [
            {
                "doc_id": sample_used[i],
                "context": match_contexts.get(
                    sample_used[i], ""
                )[:500],
            }
            for i in fp_indices[:6]
            if i < len(sample_used)
        ]
        return prec, {
            "n_checked": len(all_ratings),
            "n_fp": len(fp_indices),
            "fp_examples": fp_examples,
            "notes": "; ".join(all_notes),
        }

    # ------------------------------------------------------------------
    # LLM rating helper
    # ------------------------------------------------------------------

    def _ask_ratings(
        self, messages: list[dict], expected_n: int
    ) -> tuple[list[int], str]:
        """Ask LLM for ratings array; parse result."""
        response_tokens = max(300, 40 * expected_n + 100)
        try:
            resp = self.llm.complete_chat(messages, max_tokens=response_tokens)
            obj = self._parse_json_object(resp)
            if obj and "ratings" in obj:
                ratings = [
                    int(r) for r in obj["ratings"]
                    if isinstance(r, (int, float))
                ]
                if len(ratings) >= expected_n - 1:
                    return ratings[:expected_n], obj.get("notes", "")
        except Exception:
            pass
        return [], ""

    # ------------------------------------------------------------------
    # Refinement
    # ------------------------------------------------------------------

    def _refine(
        self, template: PatternTemplate, details: dict
    ) -> dict | None:
        """Ask the LLM to revise the regex using text from missed documents."""
        prec_d = details.get("precision_details", {})
        rec_d = details.get("recall_details", {})

        # --- False positive evidence ---
        fp_section = ""
        fps = prec_d.get("fp_examples", [])
        if fps:
            fp_section = (
                "FALSE POSITIVES (regex matched but shouldn't have):\n"
            )
            for fp in fps[:4]:
                ctx = fp.get("context", "")[:500]
                fp_section += (
                    f"  Doc: {fp.get('doc_id', '?')[:50]}\n"
                    f"    {ctx}\n\n"
                )

        # --- False negative evidence (real document text) ---
        fn_section = ""
        fns = rec_d.get("fn_examples", [])
        n_partial = details.get("partial_matches", 0)
        if fns:
            fn_section = (
                "FALSE NEGATIVES / CAPTURE FAILURES — documents where "
                "either the regex didn't match at all, OR the regex "
                "matched but all named capture groups were EMPTY "
                f"({n_partial} partial matches with empty groups). "
                "Study each region carefully; it shows the EXACT "
                "formatting variant your regex needs to handle. "
                "Pay special attention to cases where the structural "
                "pattern exists but names/values appear BEFORE or "
                "OUTSIDE the capture groups:\n\n"
            )
            # Budget: fit evidence within the LLM context
            evidence_budget = self._doc_budget_for_request(2000)
            per_fn = max(800, evidence_budget // max(len(fns), 1))

            for fn in fns:
                fn_section += f"  File: {fn.get('filename', '?')}\n"
                for ri, region in enumerate(fn.get("regions", [])):
                    truncated = region[:per_fn]
                    fn_section += (
                        f"  --- region {ri + 1} "
                        f"(of {fn.get('n_total_occurrences', '?')} "
                        f"in this doc) ---\n"
                        f"    {truncated}\n\n"
                    )

        prec_val = details.get("precision", 0)
        rec_val = details.get("recall", 0)
        n_fn = details.get("n_false_negatives", 0)

        if prec_val < self.target_precision and rec_val < self.target_recall:
            priority = (
                "PRIORITY: Both precision and recall are too low. "
                "The regex is both too loose (false positives) and too "
                "narrow (missing documents). Study the false-negative "
                "regions below character by character."
            )
        elif rec_val < self.target_recall:
            n_partial = details.get("partial_matches", 0)
            partial_note = ""
            if n_partial:
                partial_note = (
                    f"  {n_partial} of these are PARTIAL MATCHES where "
                    f"the regex fires but capture groups are empty — "
                    f"the structure exists in a variant format the "
                    f"groups don't capture. "
                )
            priority = (
                f"PRIORITY: Precision is OK but RECALL IS TOO LOW — "
                f"{n_fn} documents are missed.{partial_note}"
                f"The regions below show the EXACT text. "
                f"Study whitespace, line breaks, capitalization, "
                f"separators, and formatting character by character. "
                f"Make the regex MORE FLEXIBLE to handle all these "
                f"variants while keeping precision."
            )
        else:
            priority = (
                "PRIORITY: Recall is OK but PRECISION IS TOO LOW — "
                "the regex matches documents it shouldn't. Study the "
                "false-positive contexts and tighten the regex."
            )

        prompt = _REFINE_PROMPT.format(
            name=template.name,
            description=template.description,
            parameters=template.parameters,
            regex_template=template.regex_template,
            scope_regex=template.scope_regex,
            match_count=details.get("match_count", "?"),
            total_checked=details.get("total_docs", "?"),
            match_rate=f"{details.get('match_rate', 0):.0%}",
            precision=f"{prec_val:.0%}",
            n_fp=prec_d.get("n_fp", 0),
            precision_n=prec_d.get("n_checked", 0),
            recall=f"{rec_val:.0%}",
            n_fn=n_fn,
            recall_n=details.get("recall_info", ""),
            priority_guidance=priority,
            fp_section=fp_section,
            fn_section=fn_section,
            semantic_notes="",
        )

        messages = [
            {"role": "system", "content": _DISCOVER_SYSTEM},
            {"role": "user", "content": prompt},
        ]

        resp = self.llm.complete_chat(messages, max_tokens=2000)
        return self._parse_json_object(resp)

    # ------------------------------------------------------------------
    # Document sampling and loading
    # ------------------------------------------------------------------

    def _diverse_sample(self, n: int) -> list[str]:
        """Sample doc IDs spread evenly across the corpus."""
        all_ids = self.doc_store.all_doc_ids()
        if len(all_ids) <= n:
            return list(all_ids)
        step = max(1, len(all_ids) // n)
        return [all_ids[i] for i in range(0, len(all_ids), step)][:n]

    def _load_full_texts(self, doc_ids: list[str]) -> list[dict]:
        """Load documents — full text for large-ctx LLMs, multi-region
        excerpts for small-ctx ones."""
        budget = self._per_doc_budget()
        results = []
        for did in doc_ids:
            try:
                text = self.doc_store.get_text(did)
                meta = self.doc_store.get_meta(did)
            except KeyError:
                continue
            excerpt = (
                text if len(text) <= budget
                else self._multiregion_excerpt(text, budget)
            )
            results.append({
                "doc_id": did,
                "filename": meta.filename,
                "text": excerpt,
                "full_length": len(text),
            })
        return results

    def _per_doc_budget(self) -> int:
        budget = self._ctx_chars // 6
        return max(2000, min(budget, 12_000))

    @staticmethod
    def _multiregion_excerpt(text: str, budget: int) -> str:
        """Beginning + middle + end of a document within budget."""
        if len(text) <= budget:
            return text
        region = budget // 3
        beg = text[:region]
        mid_start = max(region, (len(text) - region) // 2)
        mid = text[mid_start : mid_start + region]
        end = text[-region:]
        return (
            f"{beg}\n[… middle of document …]\n"
            f"{mid}\n[… end of document …]\n{end}"
        )

    # ------------------------------------------------------------------
    # LLM discovery prompts
    # ------------------------------------------------------------------

    def _doc_budget_for_request(self, response_tokens: int) -> int:
        """How many chars of document text fit in one LLM call."""
        overhead_tokens = 1500  # system prompt + framing
        input_budget_tokens = self._ctx_tokens - response_tokens - overhead_tokens
        return max(4000, int(input_budget_tokens * _chars_per_token()))

    def _discover_ctx_aware(
        self, sample_docs: list[dict], verbose: bool,
    ) -> list[dict]:
        """Two-round discovery with automatic chunking based on context length."""
        response_tokens = min(4000, self._ctx_tokens // 4)
        doc_char_budget = self._doc_budget_for_request(response_tokens)

        if verbose:
            print(
                f"  Doc budget per call: ~{doc_char_budget:,} chars "
                f"(response: {response_tokens} tokens)",
                flush=True,
            )

        if verbose:
            print("  Round 1: structural patterns...", flush=True)
        r1 = self._llm_discover_chunked(
            sample_docs, _DISCOVER_SYSTEM, doc_char_budget, response_tokens,
            verbose,
        )
        if verbose:
            print(f"  Round 1: {len(r1)} patterns", flush=True)

        if verbose:
            print("  Round 2: extraction patterns (new sample)...", flush=True)
        ids2 = self._diverse_sample(self.sample_size)
        docs2 = self._load_full_texts(ids2)
        r2 = self._llm_discover_chunked(
            docs2, _DISCOVER_EXTRACTION, doc_char_budget, response_tokens,
            verbose,
        )
        if verbose:
            print(f"  Round 2: {len(r2)} patterns", flush=True)

        return r1 + r2

    def _llm_discover_chunked(
        self,
        docs: list[dict],
        system: str,
        doc_char_budget: int,
        response_tokens: int,
        verbose: bool,
    ) -> list[dict]:
        """Send documents to the LLM in chunks that fit the context window."""
        chunks = self._make_chunks(docs, doc_char_budget)

        if verbose and len(chunks) > 1:
            print(
                f"    Split into {len(chunks)} chunks "
                f"(~{doc_char_budget:,} chars each)",
                flush=True,
            )

        accumulated: list[dict] = []
        for ci, chunk in enumerate(chunks):
            prior = ""
            if accumulated:
                names = [p.get("name", "?") for p in accumulated]
                prior = (
                    f"\nPatterns found so far: {', '.join(names)}. "
                    f"Focus on NEW patterns.\n"
                )
            doc_block = self._format_doc_block(chunk)
            messages = [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": (
                        f"Corpus: {self.doc_store.doc_count()} documents.\n"
                        f"{prior}\nAnalyze ALL parts of these documents and "
                        f"identify recurring structural patterns "
                        f"expressible as parameterized regex templates.\n\n"
                        f"{doc_block}"
                    ),
                },
            ]
            resp = self.llm.complete_chat(messages, max_tokens=response_tokens)
            batch = self._parse_json_array(resp)
            accumulated.extend(batch)
            if verbose and len(chunks) > 1:
                print(
                    f"    Chunk {ci+1}/{len(chunks)}: "
                    f"+{len(batch)} (total {len(accumulated)})",
                    flush=True,
                )
        return accumulated

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fix_lazy_block_captures(regex: str) -> str:
        """Remove lazy ? from block-capture quantifiers.

        LLMs persistently write [\\s\\S]{0,N}? despite instructions not to.
        The lazy ? makes the capture nearly empty; greedy is always correct
        for block captures bounded by lookaheads or section terminators.
        """
        return re.sub(
            r'(\[\\s\\S\]\{0,\d+\})\?', r'\1',
            re.sub(r'(\[\\s\\S\]\{\d+,\d+\})\?', r'\1', regex),
        )

    def _raw_to_template(self, raw: dict) -> PatternTemplate | None:
        try:
            name = raw.get("name", "")
            if not name or not raw.get("regex_template"):
                return None
            params_raw = raw.get("parameters", [])
            params: list[str] = []
            for p in params_raw:
                if isinstance(p, str):
                    params.append(p)
                elif isinstance(p, dict) and "name" in p:
                    params.append(str(p["name"]))
                else:
                    params.append(str(p))
            regex_template = self._fix_lazy_block_captures(
                raw["regex_template"],
            )
            test_regex = regex_template
            for p in params:
                test_regex = test_regex.replace(f"{{{p}}}", "test")
            compiled = re.compile(test_regex, re.IGNORECASE | re.DOTALL)

            # Auto-wrap: if the LLM produced a valid regex with zero
            # named groups, wrap the entire match in (?P<content>...).
            if not compiled.groupindex:
                regex_template = f"(?P<content>{regex_template})"
                if "content" not in params:
                    params.append("content")
                re.compile(
                    regex_template.replace("{content}", "test"),
                    re.IGNORECASE | re.DOTALL,
                )

            return PatternTemplate(
                name=name,
                description=raw.get("description", ""),
                parameters=params,
                regex_template=regex_template,
                scope_regex=raw.get("scope_regex", ""),
                output_type=raw.get("output_type", "extract"),
            )
        except (re.error, KeyError, TypeError):
            return None

    @staticmethod
    def _format_doc_block(docs: list[dict]) -> str:
        parts = []
        for i, doc in enumerate(docs):
            parts.append(
                f"\n--- DOCUMENT {i+1}: {doc['filename']} "
                f"({doc['full_length']} chars) ---\n{doc['text']}\n"
            )
        return "".join(parts)

    @staticmethod
    def _make_chunks(
        docs: list[dict], max_chars: int
    ) -> list[list[dict]]:
        chunks: list[list[dict]] = []
        current: list[dict] = []
        current_chars = 0
        for doc in docs:
            c = len(doc["text"]) + 100
            if current and current_chars + c > max_chars:
                chunks.append(current)
                current = []
                current_chars = 0
            current.append(doc)
            current_chars += c
        if current:
            chunks.append(current)
        return chunks

    @staticmethod
    def _parse_json_array(text: str) -> list[dict]:
        text = text.strip()
        start = text.find("[")
        if start == -1:
            return []
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        return []
        return []

    @staticmethod
    def _parse_json_object(text: str) -> dict | None:
        text = text.strip()
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        return None
        return None
