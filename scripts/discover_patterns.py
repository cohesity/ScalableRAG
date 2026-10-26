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

"""Discover parameterized regex patterns from a corpus.

Reads documents, uses an LLM to identify recurring structural patterns,
builds parameterized regex templates, and validates them.

Usage:
    python scripts/discover_patterns.py \
        --store /path/to/index \
        --output patterns.json

Environment variables: LLM credentials as in scripts/query.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scalablerag.llm.factory import make_llm
from scalablerag.storage.doc_store import DocStore
from scalablerag.auto_patterns.discovery import PatternDiscovery


def main():
    parser = argparse.ArgumentParser(
        description="Discover parameterized regex patterns from a corpus"
    )
    parser.add_argument("--store", required=True, help="Indexed corpus path")
    parser.add_argument("--packed-database", default=None)
    parser.add_argument("--corpus-limit", type=int, default=None)
    parser.add_argument("--output", default="patterns.json", help="Output file")
    parser.add_argument("--llm-provider", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--sample-size", type=int, default=20,
                        help="Number of docs to sample for discovery")
    parser.add_argument("--validation-sample", type=int, default=50,
                        help="Number of docs for validation")
    parser.add_argument("--max-refinements", type=int, default=3)
    parser.add_argument("--target-score", type=float, default=0.90)

    args = parser.parse_args()

    store_path = Path(args.store)
    print("Loading corpus...")
    if args.packed_database:
        if not args.corpus_limit:
            raise SystemExit("--packed-database requires --corpus-limit")
        from scalablerag.storage.packed_store import PackedDocStore
        doc_store = PackedDocStore(args.packed_database, args.corpus_limit)
    else:
        doc_store = DocStore(store_path / "docs")
    print(f"  {doc_store.doc_count()} docs")

    print("Creating LLM...")
    llm = make_llm(
        provider=args.llm_provider,
        model=args.llm_model,
        total_tokens=5_000_000,
    )

    discovery = PatternDiscovery(
        llm=llm,
        doc_store=doc_store,
        sample_size=args.sample_size,
        precision_sample=args.validation_sample,
        max_refinement_rounds=args.max_refinements,
        target_precision=args.target_score,
        target_recall=args.target_score,
    )

    library = discovery.discover(verbose=True)
    library.save(args.output)
    diagnostics = Path(args.output).with_suffix(".discovery.json")
    diagnostics.write_text(
        json.dumps(discovery.discovery_diagnostics, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(library)} patterns to {args.output}")
    print(f"Saved discovery diagnostics to {diagnostics}")


if __name__ == "__main__":
    main()
