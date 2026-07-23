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

"""LLM-based pattern curation: score patterns for question-answering utility.

After discovery produces candidate patterns (especially from the field scanner),
this module asks an LLM to score each pattern's usefulness for answering questions
about the corpus. Batching adapts to the LLM context window.

Flow:
  1. Gather per-pattern statistics from the ExtractionIndex (or directly
     from field-scanner metadata): match count, value diversity, samples.
  2. Send patterns to the LLM in batches, asking for 0–10 scores.
  3. Return the curated list: patterns that scored above a natural quality
     threshold, annotated with the LLM's reasoning and suggested use cases.
"""

from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass, field, asdict
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..llm.base import BaseLLM
    from ..storage.doc_store import DocStore
    from .extraction_index import ExtractionIndex, PatternIndex
    from .patterns import PatternLibrary, PatternTemplate

# ── Scoring prompt ─────────────────────────────────────────────────────

_CURATION_SYSTEM = textwrap.dedent("""\
    You are evaluating automatically discovered data patterns from a document
    corpus.  Your task: score each pattern on how useful it would be for
    answering questions about the documents.

    You do NOT know what the questions will be.  Judge general utility:
    would someone researching these documents plausibly ask a question
    that this pattern helps answer?

    Scoring guide (integer 0–10):
      10  Extracts highly specific, commonly queried info (names, dates,
          monetary amounts, locations, identifiers, statuses).
       7  Extracts structured data likely to appear in questions.
       4  Moderately useful — might help for some questions.
       2  Low utility — mostly boilerplate, formatting, or rarely queried.
       0  Noise — uninformative or redundant.

    Respond with a JSON array.  Each element:
      {"name": "<pattern_name>", "score": <int 0-10>, "reasoning": "<1 sentence>", "use_case": "<when to use, or 'none'>"}

    Rules:
    - Judge ONLY by general question-answering utility.
    - Diverse, specific extracted values (proper nouns, amounts, dates)
      are more useful than generic or constant text.
    - Patterns that cover many documents AND have diverse values are
      especially valuable.
    - Be decisive: most patterns in a large batch will be low-value.
      Do not be generous — a score of 5+ means genuinely useful.
    - Output ONLY the JSON array, no other text.""")


# ── Data structures ────────────────────────────────────────────────────

@dataclass
class PatternScore:
    """Result of LLM curation for a single pattern."""
    name: str
    score: int
    reasoning: str
    use_case: str
    # Original metadata carried through for downstream use
    description: str = ""
    match_count: int = 0
    total_docs: int = 0
    value_diversity: int = 0
    sample_values: list[str] = field(default_factory=list)
    source: str = ""  # "field_scanner" or "llm_validated"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CurationResult:
    """Full result of a curation pass."""
    all_scores: list[PatternScore]
    kept: list[PatternScore]
    threshold: int
    total_patterns: int
    total_kept: int
    batch_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "total_patterns": self.total_patterns,
            "total_kept": self.total_kept,
            "batch_count": self.batch_count,
            "kept": [s.to_dict() for s in self.kept],
            "all_scores": [s.to_dict() for s in self.all_scores],
        }

    def save(self, path: str) -> None:
        from pathlib import Path as _P
        p = _P(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str) -> CurationResult:
        from pathlib import Path as _P
        data = json.loads(_P(path).read_text())
        all_scores = [PatternScore(**s) for s in data["all_scores"]]
        kept = [PatternScore(**s) for s in data["kept"]]
        return cls(
            all_scores=all_scores,
            kept=kept,
            threshold=data["threshold"],
            total_patterns=data["total_patterns"],
            total_kept=data["total_kept"],
            batch_count=data["batch_count"],
        )


# ── Statistics helpers ─────────────────────────────────────────────────

def _pattern_stats_from_index(
    pi: "PatternIndex",
) -> dict[str, Any]:
    """Extract statistics for a single pattern from the ExtractionIndex."""
    n_match = len(pi.matching_doc_ids)

    # Collect sample values from all parameters
    samples: list[str] = []
    total_unique = 0
    for param, vals in (pi.unique_values or {}).items():
        total_unique += len(vals)
        samples.extend(vals[:5])
    samples = samples[:10]

    return {
        "name": pi.pattern_name,
        "description": pi.description,
        "match_count": n_match,
        "total_docs": pi.total_docs,
        "match_pct": round(n_match * 100 / max(1, pi.total_docs), 1),
        "value_diversity": total_unique,
        "sample_values": samples,
        "parameters": pi.parameters,
    }


def _pattern_stats_from_template(
    template: "PatternTemplate",
) -> dict[str, Any]:
    """Extract statistics from a PatternTemplate (field-scanner metadata).

    For LLM-validated patterns, validation_details uses different keys
    (match_count/total_docs/precision/recall) compared to field-scanner
    patterns (doc_count/corpus_size/value_variety).  We unify both.
    """
    vd = template.validation_details or {}
    match_count = vd.get("doc_count", 0) or vd.get("match_count", 0) or 0
    total_docs = vd.get("corpus_size", 0) or vd.get("total_docs", 0) or 0
    coverage_pct = vd.get("coverage_pct", 0)
    if not coverage_pct and total_docs:
        coverage_pct = round(match_count * 100 / max(1, total_docs), 1)
    value_variety = vd.get("value_variety", 0)

    # For LLM-validated patterns with named parameters but no value_variety,
    # use the parameter count as a proxy: each named capture extracts
    # distinct information.
    source = vd.get("source", "")
    if not value_variety and template.parameters and source != "field_scanner":
        value_variety = -1  # sentinel: "has extractable fields, count unknown"

    return {
        "name": template.name,
        "description": template.description,
        "match_count": match_count,
        "total_docs": total_docs,
        "match_pct": coverage_pct,
        "value_diversity": value_variety,
        "sample_values": [],
        "parameters": template.parameters,
        "source": source,
    }


def gather_pattern_stats(
    library: "PatternLibrary",
    extraction_index: "ExtractionIndex | None" = None,
) -> list[dict[str, Any]]:
    """Gather statistics for all patterns from the best available source."""
    stats: list[dict[str, Any]] = []
    for template in library.all_templates():
        if extraction_index:
            pi = extraction_index.get_pattern(template.name)
            if pi:
                s = _pattern_stats_from_index(pi)
                s["source"] = (template.validation_details or {}).get(
                    "source", "llm_validated"
                )
                stats.append(s)
                continue
        s = _pattern_stats_from_template(template)
        if not s.get("source"):
            s["source"] = "llm_validated"
        stats.append(s)
    return stats


# ── Prompt construction ────────────────────────────────────────────────

def _format_pattern_for_prompt(stat: dict[str, Any]) -> str:
    """Format a single pattern's statistics for the LLM prompt."""
    lines = [f"- **{stat['name']}**"]
    lines.append(f"  Description: {stat['description']}")
    lines.append(
        f"  Coverage: {stat['match_count']}/{stat['total_docs']} docs "
        f"({stat['match_pct']}%)"
    )
    vd = stat.get("value_diversity", 0)
    if vd and vd > 0:
        lines.append(f"  Distinct values: {vd}")
    if stat.get("sample_values"):
        vals = stat["sample_values"][:6]
        lines.append(f"  Sample values: {vals}")
    if stat.get("parameters"):
        lines.append(f"  Named captures: {stat['parameters']}")
    return "\n".join(lines)


def _build_batch_prompt(batch: list[dict[str, Any]]) -> str:
    """Build the user prompt for a single batch of patterns."""
    header = (
        f"Below are {len(batch)} automatically discovered patterns from a "
        f"document corpus.  Score each for question-answering utility (0–10).\n\n"
    )
    body = "\n\n".join(_format_pattern_for_prompt(s) for s in batch)
    return header + body


# ── Core curation logic ────────────────────────────────────────────────

def _estimate_tokens_per_pattern(stats: list[dict[str, Any]]) -> float:
    """Rough token estimate per pattern in the prompt."""
    if not stats:
        return 200
    total_chars = sum(len(_format_pattern_for_prompt(s)) for s in stats[:20])
    avg_chars = total_chars / min(20, len(stats))
    return avg_chars / 3.5 + 60  # +60 for response tokens per pattern


def _compute_batch_size(
    n_patterns: int,
    tokens_per_pattern: float,
    context_length: int,
    max_completion: int,
) -> int:
    """Compute optimal batch size for the LLM's context window.

    The binding constraint is usually the response: each scored pattern
    needs ~100–150 tokens in the JSON output (name, score, reasoning,
    use_case).  We target using at most 80% of the completion budget
    to leave margin for JSON framing.
    """
    prompt_overhead = 800  # system prompt + framing
    response_per_pattern = 130  # empirical: ~130 tokens per scored pattern

    # Available input tokens for pattern descriptions
    available_input = context_length - max_completion - prompt_overhead
    input_capacity = max(1, int(available_input / tokens_per_pattern))

    # Response capacity: how many patterns fit in 80% of completion budget
    usable_completion = int(max_completion * 0.80)
    response_capacity = max(1, int(usable_completion / response_per_pattern))

    # Hard cap: keep batches manageable for reliable JSON output
    batch = min(input_capacity, response_capacity, n_patterns, 150)
    return max(1, batch)


def _parse_scores(response: str, expected_names: set[str]) -> list[dict]:
    """Parse LLM JSON response into score dicts, tolerating minor issues."""
    text = response.strip()
    # Find JSON array boundaries
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        arr = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        # Try line-by-line recovery
        arr = []
        for line in text[start : end + 1].split("\n"):
            line = line.strip().rstrip(",")
            if line.startswith("{"):
                try:
                    arr.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    results = []
    for item in arr:
        if not isinstance(item, dict):
            continue
        name = item.get("name", "")
        score = item.get("score")
        if isinstance(score, (int, float)):
            score = max(0, min(10, int(round(score))))
        else:
            score = 0
        results.append({
            "name": name,
            "score": score,
            "reasoning": str(item.get("reasoning", "")),
            "use_case": str(item.get("use_case", "none")),
        })
    return results


def curate_patterns(
    stats: list[dict[str, Any]],
    llm: "BaseLLM",
    *,
    threshold: int = 5,
    batch_size: int | None = None,
    verbose: bool = True,
) -> CurationResult:
    """Score all patterns via LLM and return curation result.

    Parameters
    ----------
    stats : list of pattern stat dicts (from gather_pattern_stats)
    llm : LLM instance for scoring
    threshold : minimum score to keep (default 5 = "genuinely useful")
    batch_size : override auto-computed batch size
    verbose : print progress
    """
    if not stats:
        return CurationResult([], [], threshold, 0, 0, 0)

    tpp = _estimate_tokens_per_pattern(stats)
    if batch_size is None:
        batch_size = _compute_batch_size(
            len(stats), tpp, llm.context_length, llm.max_completion_tokens,
        )
    if verbose:
        print(
            f"[curate] {len(stats)} patterns, batch_size={batch_size}, "
            f"~{tpp:.0f} tokens/pattern, threshold={threshold}",
            flush=True,
        )

    all_scores: list[PatternScore] = []
    batch_count = 0
    stat_by_name = {s["name"]: s for s in stats}

    for i in range(0, len(stats), batch_size):
        batch = stats[i : i + batch_size]
        batch_count += 1
        if verbose:
            print(
                f"[curate] Batch {batch_count}: patterns {i+1}–{i+len(batch)} "
                f"of {len(stats)}",
                flush=True,
            )

        prompt = _build_batch_prompt(batch)
        expected = {s["name"] for s in batch}

        max_tokens = min(
            llm.max_completion_tokens,
            max(4096, len(batch) * 150),
        )

        try:
            response = llm.complete_chat(
                [
                    {"role": "system", "content": _CURATION_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
            )
        except Exception as e:
            if verbose:
                print(f"  [curate] LLM call failed: {e}", flush=True)
            response = ""

        parsed = _parse_scores(response, expected)

        # For structured output, scores might be nested under "scores" key
        if not parsed:
            try:
                obj = json.loads(response.strip())
                if isinstance(obj, dict) and "scores" in obj:
                    parsed = obj["scores"]
            except (json.JSONDecodeError, TypeError):
                pass

        scored_names = set()
        for item in parsed:
            name = item.get("name", "") if isinstance(item, dict) else ""
            if not name:
                continue
            scored_names.add(name)
            orig = stat_by_name.get(name, {})
            ps = PatternScore(
                name=name,
                score=item.get("score", 0) if isinstance(item, dict) else 0,
                reasoning=item.get("reasoning", "") if isinstance(item, dict) else "",
                use_case=item.get("use_case", "none") if isinstance(item, dict) else "none",
                description=orig.get("description", ""),
                match_count=orig.get("match_count", 0),
                total_docs=orig.get("total_docs", 0),
                value_diversity=orig.get("value_diversity", 0),
                sample_values=orig.get("sample_values", []),
                source=orig.get("source", ""),
            )
            all_scores.append(ps)

        missed = expected - scored_names
        if missed and verbose:
            print(
                f"  [curate] WARNING: {len(missed)} patterns not scored "
                f"in batch {batch_count}: {list(missed)[:5]}...",
                flush=True,
            )
        # Assign score 0 to missed patterns so they don't silently pass
        for name in missed:
            orig = stat_by_name.get(name, {})
            all_scores.append(PatternScore(
                name=name, score=0, reasoning="Not scored by LLM",
                use_case="none",
                description=orig.get("description", ""),
                match_count=orig.get("match_count", 0),
                total_docs=orig.get("total_docs", 0),
                value_diversity=orig.get("value_diversity", 0),
                sample_values=orig.get("sample_values", []),
                source=orig.get("source", ""),
            ))

        if verbose:
            above = sum(1 for it in parsed
                        if isinstance(it, dict) and it.get("score", 0) >= threshold)
            print(
                f"  [curate] Batch {batch_count}: {len(parsed)} scored, "
                f"{above} above threshold ({threshold})",
                flush=True,
            )

    all_scores.sort(key=lambda s: -s.score)
    kept = [s for s in all_scores if s.score >= threshold]

    if verbose:
        print(
            f"[curate] Done: {len(all_scores)} scored, "
            f"{len(kept)} kept (score >= {threshold})",
            flush=True,
        )
        if kept:
            print("[curate] Top patterns:", flush=True)
            for s in kept[:15]:
                print(
                    f"  {s.score:2d}  {s.name}: {s.reasoning}",
                    flush=True,
                )

    return CurationResult(
        all_scores=all_scores,
        kept=kept,
        threshold=threshold,
        total_patterns=len(stats),
        total_kept=len(kept),
        batch_count=batch_count,
    )


# ── Gap discovery — find important concepts not covered by regex patterns ──

_GAP_SYSTEM = textwrap.dedent("""\
    You are analyzing a document corpus.  A regex-based scanner already
    found Label: Value fields (dates, names, IDs, etc.).  Your task:
    identify SPECIFIC EXTRACTABLE VALUES embedded in prose that no
    Label: Value pattern can capture.

    GOOD gaps — specific values that VARY across documents:
    - Numeric quantities (amounts, counts, durations) buried in
      narrative text rather than labeled fields
    - Named entities (people, organizations, locations) that appear
      in running prose and vary from document to document
    - Dates, deadlines, or time periods mentioned in sentences
    - Categories, types, or classifications implied by context
    - Specific thresholds, rates, or percentages in body text

    BAD gaps — do NOT suggest these:
    - Broad sections or headings that exist in every document
    - General topics or themes common to the whole corpus
    - Things that are the same boilerplate text across all documents

    The key test: would the ANSWER to a question about this concept
    be a specific value (a number, name, date, category) that differs
    from document to document?

    For each concept, provide:
      {"name": "<short_snake_case>",
       "description": "<what specific value this extracts, 1 sentence>",
       "extraction_hint": "<how to locate and extract the specific value>",
       "search_keywords": ["keyword1", "keyword2", ...]}

    search_keywords should be 2-5 SPECIFIC phrases (not single common
    words) that co-occur with the actual value in the text.

    Output a JSON array.  Prefer quality over quantity: 3-12 concepts
    that genuinely represent extractable, varying values.
    Output ONLY the JSON array.""")


def discover_gaps(
    kept_patterns: list[PatternScore],
    doc_samples: list[str],
    llm: "BaseLLM",
    *,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Discover important concepts not covered by existing patterns.

    Uses 1 LLM call.  ``doc_samples`` should be 5-10 representative
    document texts (or large excerpts) so the LLM can see what the
    corpus actually contains beyond Label:Value fields.
    """
    if verbose:
        print("[gaps] Discovering concepts not covered by regex patterns...",
              flush=True)

    pattern_summary = "\n".join(
        f"  • {p.name}: {p.description}" for p in kept_patterns[:80]
    )

    sample_text = ""
    budget = int(llm.context_length * 2.5)  # chars budget for samples
    for i, s in enumerate(doc_samples):
        chunk = s[:budget // max(1, len(doc_samples))]
        sample_text += f"\n--- Document sample {i+1} ---\n{chunk}\n"

    user_prompt = (
        f"The following {len(kept_patterns)} structural patterns were already "
        f"discovered by a regex scanner.  They capture Label: Value fields:\n\n"
        f"{pattern_summary}\n\n"
        f"Below are sample documents from the corpus:\n{sample_text}\n\n"
        f"What important concepts in these documents are NOT captured by "
        f"any of the above patterns?  Focus on information embedded in prose "
        f"paragraphs, clauses, or narrative sections."
    )

    max_tokens = min(llm.max_completion_tokens, 4096)
    try:
        response = llm.complete_chat(
            [{"role": "system", "content": _GAP_SYSTEM},
             {"role": "user", "content": user_prompt}],
            max_tokens=max_tokens,
        )
    except Exception as e:
        if verbose:
            print(f"[gaps] LLM call failed: {e}", flush=True)
        return []

    # Parse JSON array from response
    text = response.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        if verbose:
            print("[gaps] Could not parse response", flush=True)
        return []
    try:
        concepts = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        if verbose:
            print("[gaps] JSON parse error", flush=True)
        return []

    results = []
    for c in concepts:
        if not isinstance(c, dict) or not c.get("name"):
            continue
        results.append({
            "name": c["name"],
            "description": str(c.get("description", "")),
            "extraction_hint": str(c.get("extraction_hint", "")),
            "search_keywords": c.get("search_keywords", []),
        })

    if verbose:
        print(f"[gaps] Discovered {len(results)} gap concepts:", flush=True)
        for r in results:
            print(f"  • {r['name']}: {r['description']}", flush=True)

    return results


def estimate_concept_coverage(
    concepts: list[dict[str, Any]],
    doc_store: "DocStore",
    *,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Estimate how many docs contain each concept via keyword matching.

    Pure string matching — zero LLM cost.  For each concept, searches
    all documents for its search_keywords and counts matches.
    """
    if TYPE_CHECKING:
        from ..storage.doc_store import DocStore as _DS

    all_ids = doc_store.all_doc_ids()
    n_docs = len(all_ids)

    # Pre-load all doc texts once
    doc_texts: dict[str, str] = {}
    for did in all_ids:
        try:
            doc_texts[did] = doc_store.get_text(did).lower()
        except (KeyError, FileNotFoundError):
            pass

    for concept in concepts:
        keywords = [k.lower() for k in concept.get("search_keywords", [])
                    if k and len(k) >= 2]
        if not keywords:
            concept["match_count"] = 0
            concept["matching_doc_ids"] = []
            continue

        matching = []
        for did, text in doc_texts.items():
            if any(kw in text for kw in keywords):
                matching.append(did)

        concept["match_count"] = len(matching)
        concept["total_docs"] = n_docs
        concept["match_pct"] = round(len(matching) * 100 / max(1, n_docs), 1)
        concept["matching_doc_ids"] = matching

        if verbose:
            print(
                f"  [{concept['name']}] {len(matching)}/{n_docs} docs "
                f"({concept['match_pct']}%) matched keywords {keywords}",
                flush=True,
            )

    return concepts


# ── Deduplication — merge semantically equivalent patterns ────────────

_DEDUP_SYSTEM = textwrap.dedent("""\
    You are merging a list of document extraction patterns.  Some patterns
    capture essentially the same information under different names or
    slightly different formats (e.g. "email" vs "email_address" vs
    "e_mail_address", or "customer_name" vs "customer_company_name").

    Group semantically equivalent patterns into clusters.  For each cluster,
    pick the BEST representative — prefer the one with highest coverage
    (most matching documents) or most descriptive name.

    IMPORTANT: Only merge patterns that truly extract the SAME information.
    Do NOT merge patterns that are related but distinct (e.g. "date_signed"
    and "effective_date" are different dates — keep them separate).

    Output a JSON array of cluster objects:
      [{"representative": "<name>", "members": ["<name1>", "<name2>", ...], "merged_description": "<combined description>"}]

    Include singleton clusters (patterns with no duplicates) too.
    Output ONLY the JSON array.""")


def deduplicate_patterns(
    patterns: list[PatternScore],
    llm: "BaseLLM",
    *,
    verbose: bool = True,
) -> list[PatternScore]:
    """Merge semantically equivalent patterns.  Uses 1 LLM call.

    Returns the deduplicated list with merged descriptions where
    appropriate.
    """
    if len(patterns) <= 1:
        return list(patterns)

    if verbose:
        print(f"[dedup] Deduplicating {len(patterns)} patterns...", flush=True)

    # Build a compact representation for the LLM
    lines = []
    for p in patterns:
        lines.append(
            f"  • {p.name} ({p.match_count}/{p.total_docs} docs): "
            f"{p.description}"
        )
    pattern_list = "\n".join(lines)

    user_prompt = (
        f"Below are {len(patterns)} curated patterns from a document corpus.\n\n"
        f"{pattern_list}\n\n"
        f"Group semantically equivalent patterns into clusters."
    )

    max_tokens = min(llm.max_completion_tokens, max(4096, len(patterns) * 80))
    try:
        response = llm.complete_chat(
            [{"role": "system", "content": _DEDUP_SYSTEM},
             {"role": "user", "content": user_prompt}],
            max_tokens=max_tokens,
        )
    except Exception as e:
        if verbose:
            print(f"[dedup] LLM call failed: {e}", flush=True)
        return list(patterns)

    # Parse response
    text = response.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        if verbose:
            print("[dedup] Could not parse response, skipping dedup", flush=True)
        return list(patterns)
    try:
        clusters = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        if verbose:
            print("[dedup] JSON parse error, skipping dedup", flush=True)
        return list(patterns)

    by_name = {p.name: p for p in patterns}
    kept: list[PatternScore] = []
    seen: set[str] = set()
    merged_count = 0

    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        rep_name = cluster.get("representative", "")
        members = cluster.get("members", [])
        merged_desc = cluster.get("merged_description", "")

        rep = by_name.get(rep_name)
        if not rep:
            # Try to find the representative among members
            for m in members:
                if m in by_name:
                    rep = by_name[m]
                    rep_name = m
                    break
        if not rep:
            continue

        if rep_name in seen:
            continue
        seen.add(rep_name)

        # If cluster has multiple members, update description and aggregate
        # the max match_count across merged patterns
        non_rep_members = [m for m in members if m != rep_name and m in by_name]
        if non_rep_members:
            merged_count += len(non_rep_members)
            if merged_desc:
                rep.description = merged_desc
            # Take the maximum coverage from all merged patterns
            for m in non_rep_members:
                seen.add(m)
                other = by_name[m]
                if other.match_count > rep.match_count:
                    rep.match_count = other.match_count
        kept.append(rep)

    # Add any patterns the LLM didn't mention (safety net)
    for p in patterns:
        if p.name not in seen:
            kept.append(p)

    if verbose:
        print(
            f"[dedup] {len(patterns)} → {len(kept)} patterns "
            f"({merged_count} merged away)",
            flush=True,
        )

    return kept


# ── Full curation pipeline ────────────────────────────────────────────

def full_curation_pipeline(
    stats: list[dict[str, Any]],
    llm: "BaseLLM",
    doc_store: "DocStore | None" = None,
    doc_samples: list[str] | None = None,
    *,
    threshold: int = 5,
    batch_size: int | None = None,
    discover_semantic_gaps: bool = True,
    deduplicate: bool = True,
    verbose: bool = True,
) -> CurationResult:
    """Run the complete curation pipeline: score → gaps → dedup.

    LLM call counts (approximate):
      - Scoring: ~N/batch_size calls (same as curate_patterns)
      - Gap discovery: 1 call (if discover_semantic_gaps=True)
      - Deduplication: 1 call (if deduplicate=True)

    Keyword coverage uses string matching only (no per-document LLM calls).
    """
    # Step 1: Score all regex patterns
    result = curate_patterns(
        stats, llm, threshold=threshold,
        batch_size=batch_size, verbose=verbose,
    )

    # Step 2: Discover semantic gaps (concepts not covered by regex)
    if discover_semantic_gaps and doc_samples and result.kept:
        gap_concepts = discover_gaps(
            result.kept, doc_samples, llm, verbose=verbose,
        )
        if gap_concepts and doc_store is not None:
            gap_concepts = estimate_concept_coverage(
                gap_concepts, doc_store, verbose=verbose,
            )

        # Filter concepts with too few matches to be meaningful, then
        # score them through the SAME LLM curation prompt used for
        # regex patterns.  No coverage-based heuristics — the LLM
        # decides what's genuinely useful for QA.
        gap_stats: list[dict[str, Any]] = []
        for gc in gap_concepts:
            mc = gc.get("match_count", 0)
            td = gc.get("total_docs", 0)
            if mc < 3:
                if verbose:
                    print(
                        f"  [gaps] Skipping '{gc['name']}': "
                        f"only {mc} docs matched keywords",
                        flush=True,
                    )
                continue
            gap_stats.append({
                "name": gc["name"],
                "description": (
                    f"{gc['description']}  "
                    f"Extraction: {gc.get('extraction_hint', 'N/A')}"
                ),
                "match_count": mc,
                "total_docs": td,
                "match_pct": round(mc * 100 / max(1, td), 1),
                "value_diversity": -1,  # unknown for prose-embedded
                "sample_values": [],
                "parameters": [],
                "source": "semantic_gap",
            })

        if gap_stats:
            if verbose:
                print(
                    f"[gaps] Scoring {len(gap_stats)} gap concepts "
                    f"through LLM curation...",
                    flush=True,
                )
            gap_result = curate_patterns(
                gap_stats, llm, threshold=threshold,
                verbose=verbose,
            )
            added = 0
            for ps in gap_result.all_scores:
                ps.source = "semantic_gap"
                # Carry the extraction hint through as use_case
                orig = next(
                    (g for g in gap_concepts if g["name"] == ps.name),
                    None,
                )
                if orig and not ps.use_case:
                    ps.use_case = orig.get("extraction_hint", "")
                result.all_scores.append(ps)
                if ps.score >= threshold:
                    result.kept.append(ps)
                    added += 1
            if verbose:
                print(
                    f"[gaps] {added} semantic patterns scored above "
                    f"threshold ({threshold})",
                    flush=True,
                )

    # Step 3: Deduplicate
    if deduplicate and len(result.kept) > 1:
        result.kept = deduplicate_patterns(
            result.kept, llm, verbose=verbose,
        )

    result.total_kept = len(result.kept)

    if verbose:
        print(
            f"\n[pipeline] Final result: {result.total_kept} curated patterns "
            f"(from {result.total_patterns} discovered)",
            flush=True,
        )

    return result
