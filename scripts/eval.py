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

"""Evaluate the retrieval agent on a questions.jsonl file.

Produces two output files:
  1. results.jsonl        — one line per question: answer, gold, compact trace
  2. {stem}_traces.jsonl  — full step-by-step trace with every observation

Usage:
    python scripts/eval.py --regime zero --store INDEX --questions Q.jsonl
    python scripts/eval.py --regime limited --store INDEX --questions Q.jsonl

Environment variables: LLM and embed credentials as in scripts/query.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scalablerag.llm.factory import make_llm
from scalablerag.storage.doc_store import DocStore
from scalablerag.run_config import (
    detect_explicit_flags,
    load_embedding_setup,
    resolve_run_config,
)
from scalablerag.retrieval.agent import RetrievalAgent


def load_questions(path: str) -> list[dict]:
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def _trace_path(output_path: Path) -> Path:
    """Derive trace filename from results filename."""
    return output_path.with_name(
        output_path.stem + "_traces" + output_path.suffix
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval agent")
    parser.add_argument("--store", required=True, help="Indexed corpus path")
    parser.add_argument("--questions", required=True, help="questions.jsonl")
    parser.add_argument("--output", default="results.jsonl", help="Output file")
    parser.add_argument(
        "--regime",
        choices=["zero", "limited"],
        default=None,
        help="zero or limited (sets embeddings, patterns, agg, BaS)",
    )
    parser.add_argument("--llm-provider", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--embed-provider", default=None)
    parser.add_argument("--embed-model", default=None)
    parser.add_argument("--embed-dir", default=None,
                        help="Embedding subdirectory name (e.g. embeddings__hf__all-MiniLM-L6-v2)")
    parser.add_argument(
        "--embeddings", action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--patterns", default=None,
                        help="Path to pattern library JSON (from discover_patterns.py)")
    parser.add_argument(
        "--discover-patterns", action=argparse.BooleanOptionalAction,
        help="Run pattern discovery before eval",
    )
    parser.add_argument(
        "--curate-patterns", action=argparse.BooleanOptionalAction,
        help="Run LLM pattern curation after discovery/loading",
    )
    parser.add_argument("--min-pattern-coverage", type=float, default=0.10,
                        help="Min corpus coverage (0-1) for curated patterns (default: 0.10)")
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--start", type=int, default=0, help="Start from question index")
    parser.add_argument("--limit", type=int, default=0, help="Max questions (0=all)")
    parser.add_argument(
        "--agg-tools", action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--branch-select", action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    cfg = resolve_run_config(args, detect_explicit_flags(sys.argv))

    if cfg.regime:
        print(f"Regime: {cfg.regime}")
    print(
        f"  embeddings={'OFF' if cfg.skip_embeddings else 'ON'}, "
        f"patterns={'discover' if cfg.discover_patterns else 'load/skip'}, "
        f"curation={'ON' if cfg.curate_patterns else 'OFF'}, "
        f"agg_tools={'ON' if cfg.use_agg_tools else 'OFF'}, "
        f"branch_select={'ON' if cfg.branch_select else 'OFF'}"
    )

    store_path = Path(args.store)
    print("Loading corpus...")
    doc_store = DocStore(store_path / "docs")

    embedding_store = None
    embedder = None
    use_embeddings = False

    if cfg.skip_embeddings:
        print("  Embeddings DISABLED")
    else:
        print("Creating embedder...")
        embedding_store, embedder, use_embeddings = load_embedding_setup(
            store_path,
            skip_embeddings=False,
            embed_provider=cfg.embed_provider,
            embed_model=cfg.embed_model,
            embed_dir=cfg.embed_dir,
        )
        if embedder is not None:
            print(f"  Embedder dim={embedder.dimension}")
        if embedding_store is not None:
            print(
                f"  Embedding store: {embedding_store.store_dir.name} "
                f"(dim={embedding_store.dimension}, docs={embedding_store.doc_count})"
            )
        elif cfg.require_embeddings:
            print("ERROR: no compatible embedding store on disk")
            sys.exit(1)
        else:
            print("  no compatible embedding store; running without embeddings")
            print("  ingest with matching embedder, or pass --embed-dir, or --regime zero")

    if cfg.require_embeddings and not use_embeddings:
        print("ERROR: embeddings required but not available")
        sys.exit(1)

    print(f"  {doc_store.doc_count()} docs, embeddings={'ON' if use_embeddings else 'OFF'}")

    print("Creating LLM...")
    llm = make_llm(
        provider=args.llm_provider,
        model=args.llm_model,
        total_tokens=10_000_000,
    )
    print(f"  Context: {llm.context_length}")

    # Pattern library + extraction index (optional preprocessing)
    pattern_library = None
    extraction_index = None
    patterns_path = args.patterns or os.environ.get("SR_PATTERNS_FILE", "")

    if not patterns_path and cfg.discover_patterns:
        from scalablerag.auto_patterns.discovery import PatternDiscovery
        from scalablerag.auto_patterns.patterns import PatternLibrary
        print("\nRunning pattern discovery (preprocessing)...")
        discovery = PatternDiscovery(llm=llm, doc_store=doc_store)
        pattern_library = discovery.discover(verbose=args.verbose)
        out_path = store_path / "patterns.json"
        # Remove stale sidecar files from a prior discovery run.
        for stale in (
            out_path.with_suffix(".extraction.json"),
            out_path.with_suffix(".curation.json"),
        ):
            if stale.exists():
                stale.unlink()
                print(f"  Removed stale {stale.name}")
        pattern_library.save(out_path)
        patterns_path = str(out_path)
        print(f"  Saved patterns to {out_path}")
    elif patterns_path:
        from scalablerag.auto_patterns.patterns import PatternLibrary
        ppath = Path(patterns_path)
        if ppath.exists():
            pattern_library = PatternLibrary.load(ppath)
            n_valid = sum(
                1 for t in pattern_library.all_templates()
                if t.validation_score >= 0.5
            )
            print(
                f"  Pattern library: {len(pattern_library)} patterns "
                f"({n_valid} validated)"
            )
        else:
            print(f"  Pattern file not found: {ppath}")

    # Pattern curation: LLM-based scoring to keep only useful patterns.
    # Run BEFORE building the extraction index so we only index the
    # curated subset (e.g. ~80 patterns instead of ~1800).
    curation_result = None
    do_curate = cfg.curate_patterns

    if pattern_library and do_curate:
        from scalablerag.auto_patterns.curation import (
            CurationResult, full_curation_pipeline, gather_pattern_stats,
        )
        import random as _random
        curation_path = (
            Path(patterns_path).with_suffix(".curation.json")
            if patterns_path else None
        )
        if curation_path and curation_path.exists():
            print("Loading pre-computed curation result...")
            curation_result = CurationResult.load(str(curation_path))
            print(
                f"  {curation_result.total_kept}/{curation_result.total_patterns} "
                f"patterns kept (score >= {curation_result.threshold})"
            )
        else:
            print("Running full curation pipeline (score → gaps → dedup)...")
            # gather_pattern_stats works without extraction_index —
            # falls back to field-scanner metadata from the templates
            stats = gather_pattern_stats(pattern_library, extraction_index=None)
            all_ids = doc_store.all_doc_ids()
            sample_ids = _random.sample(all_ids, min(8, len(all_ids)))
            doc_samples = []
            for sid in sample_ids:
                try:
                    doc_samples.append(doc_store.get_text(sid)[:8000])
                except (KeyError, FileNotFoundError):
                    pass
            curation_result = full_curation_pipeline(
                stats, llm,
                doc_store=doc_store,
                doc_samples=doc_samples,
                verbose=args.verbose,
            )
            if curation_path:
                curation_result.save(str(curation_path))
                print(f"  Saved curation to {curation_path}")

    # Apply min-coverage filter to curated patterns so we don't overwhelm
    # the agent prompt with dozens of low-coverage patterns.
    min_cov = args.min_pattern_coverage
    if curation_result and curation_result.kept and min_cov > 0:
        before = len(curation_result.kept)
        curation_result.kept = [
            p for p in curation_result.kept
            if p.total_docs == 0
            or (p.match_count / max(p.total_docs, 1)) >= min_cov
        ]
        after = len(curation_result.kept)
        curation_result.total_kept = after
        if after < before:
            print(
                f"  Coverage filter (>={min_cov:.0%}): "
                f"{before} → {after} patterns"
            )

    # Build extraction index ONLY for curated patterns (if curation ran).
    # This is much faster: ~80 patterns vs ~1800 raw patterns.
    if pattern_library:
        from scalablerag.auto_patterns.extraction_index import (
            ExtractionIndex, build_extraction_index,
        )
        from scalablerag.auto_patterns.patterns import PatternLibrary as _PL

        idx_path = Path(patterns_path).with_suffix(".extraction.json") if patterns_path else None

        # Determine which pattern names the index MUST contain.
        if curation_result and curation_result.kept:
            _expected_names = {s.name for s in curation_result.kept}
        else:
            _expected_names = {t.name for t in pattern_library.all_templates()}

        if idx_path and idx_path.exists():
            print("Loading pre-computed extraction index...")
            extraction_index = ExtractionIndex.load(idx_path)
            print(f"  {len(extraction_index)} patterns indexed")

            # Rebuild if cached index doesn't match current pattern set.
            actual_names = set(extraction_index.pattern_names)
            missing = _expected_names - actual_names
            if missing:
                print(
                    f"  extraction index missing {len(missing)}/{len(_expected_names)} "
                    f"patterns; rebuilding"
                )
                extraction_index = None
                idx_path.unlink()

        if extraction_index is None:
            lib_to_index = pattern_library
            if curation_result and curation_result.kept:
                kept_names = {s.name for s in curation_result.kept}
                kept_templates = [
                    t for t in pattern_library.all_templates()
                    if t.name in kept_names
                ]
                if kept_templates:
                    lib_to_index = _PL()
                    for t in kept_templates:
                        lib_to_index.add(t)
                    print(
                        f"Building extraction index for {len(kept_templates)} "
                        f"curated patterns (skipping {len(pattern_library) - len(kept_templates)} "
                        f"low-scoring patterns)..."
                    )
                else:
                    print("Building extraction index (one-time precomputation)...")
            else:
                print("Building extraction index (one-time precomputation)...")

            extraction_index = build_extraction_index(
                lib_to_index, doc_store, verbose=args.verbose,
            )
            if idx_path:
                extraction_index.save(idx_path)
                print(f"  Saved extraction index to {idx_path}")

    agent = RetrievalAgent(
        llm=llm,
        doc_store=doc_store,
        embedding_store=embedding_store,
        embedder=embedder,
        max_steps=args.max_steps,
        use_embeddings=use_embeddings,
        branch_select=cfg.branch_select,
        pattern_library=pattern_library,
        extraction_index=extraction_index,
        curation_result=curation_result,
        use_agg_tools=cfg.use_agg_tools,
    )

    questions = load_questions(args.questions)
    print(f"\nLoaded {len(questions)} questions")

    if args.start > 0:
        questions = questions[args.start:]
    if args.limit > 0:
        questions = questions[: args.limit]
    print(f"Running {len(questions)} questions (start={args.start})\n")

    output_path = Path(args.output)
    trace_output = _trace_path(output_path)

    # Resume support: skip already-answered questions
    existing_ids = set()
    if output_path.exists():
        with open(output_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        existing_ids.add(obj.get("question_id"))
                    except Exception:
                        pass
        print(f"Resuming: {len(existing_ids)} questions already answered\n")

    total = len(questions)
    for i, q in enumerate(questions):
        qid = q.get("question_id", str(i))
        if qid in existing_ids:
            continue

        question = q["question"]
        golden = (
            q.get("golden_answer")
            or q.get("gold")
            or q.get("gold_answer")
            or q.get("answer")
            or ""
        )
        qtype = q.get("question_type", "")

        print(f"\n{'='*70}")
        print(f"[{i+1}/{total}] Q{qid} ({qtype}): {question}")
        print(f"  Golden: {golden[:200]}")
        print("=" * 70)

        try:
            result = agent.query(question, verbose=args.verbose)
        except Exception as e:
            print(f"  ERROR: {e}")
            result = {
                "answer": f"Error: {e}",
                "steps": 0,
                "tool_calls": [],
                "trace": [],
                "elapsed_s": 0,
            }

        answer_raw = result.get("answer", "")
        n_steps = result.get("steps", 0)
        forced = result.get("forced", False)
        elapsed = result.get("elapsed_s", 0)

        # Compact agent trace for results.jsonl
        compact_steps = []
        for t in result.get("trace", []):
            entry: dict = {"step": t.get("step", 0), "type": t.get("event", "")}
            if t.get("thought"):
                entry["thought"] = t["thought"][:500]
            if t.get("tool"):
                entry["tool"] = t["tool"]
                entry["args"] = t.get("args", {})
            if t.get("result"):
                entry["result_preview"] = t["result"][:1000]
                if len(t["result"]) > 1000:
                    entry["truncated"] = True
            if t.get("branch"):
                entry["branch"] = t["branch"]
            if t.get("original_proposal"):
                entry["original_proposal"] = t["original_proposal"]
            compact_steps.append(entry)

        agent_trace = {
            "n_steps": n_steps,
            "strategy": "funnel",
            "forced_answer": forced,
            "steps": compact_steps,
        }

        record = {
            "question_id": qid,
            "question_type": qtype,
            "question": question,
            "golden_answer": golden,
            "answer_raw": answer_raw,
            "elapsed_s": elapsed,
            "agent_trace": agent_trace,
        }
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        # Trace file: full step-by-step log with complete tool results
        trace_record = {
            "question_id": qid,
            "question_type": qtype,
            "question": question,
            "golden_answer": golden,
            "answer_raw": answer_raw,
            "steps": n_steps,
            "elapsed_s": elapsed,
            "forced": forced,
            "trace": result.get("trace", []),
        }
        with open(trace_output, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace_record) + "\n")

        print(f"\n  Agent: {answer_raw[:300]}")
        print(f"  Steps: {n_steps}, Time: {elapsed}s")

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    if output_path.exists():
        results = []
        with open(output_path, "r") as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))

        n = len(results)
        print(f"Total questions: {n}")
        total_time = sum(r.get("elapsed_s", 0) for r in results)
        total_steps = sum(
            r.get("agent_trace", {}).get("n_steps", r.get("steps", 0))
            for r in results
        )
        forced = sum(
            1 for r in results
            if r.get("agent_trace", {}).get("forced_answer", r.get("forced"))
        )
        print(f"Total time: {total_time:.1f}s")
        print(f"Avg time: {total_time/max(1,n):.1f}s")
        print(f"Avg steps: {total_steps/max(1,n):.1f}")
        print(f"Forced answers: {forced}")

        by_type: dict[str, list] = {}
        for r in results:
            t = r.get("question_type", "unknown")
            by_type.setdefault(t, []).append(r)

        print(f"\nBy question type:")
        for t, rs in sorted(by_type.items()):
            avg_t = sum(r.get("elapsed_s", 0) for r in rs) / len(rs)
            avg_s = sum(
                r.get("agent_trace", {}).get("n_steps", r.get("steps", 0))
                for r in rs
            ) / len(rs)
            print(f"  {t}: {len(rs)} questions, avg {avg_t:.1f}s, avg {avg_s:.1f} steps")

    print(f"\nResults: {output_path}")
    print(f"Traces:  {trace_output}")


if __name__ == "__main__":
    main()
