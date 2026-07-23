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

"""Ingest a document corpus: load files, create embeddings, persist index.

Usage:
    python scripts/ingest.py --source /path/to/docs --store /path/to/store

Environment variables:
    SR_EMBED_PROVIDER   huggingface | openai | azure (default: huggingface)
    SR_EMBED_MODEL      Model name (default depends on provider)
    OPENAI_API_KEY      Required for openai/azure embeddings
    SR_AZURE_ENDPOINT   Required when SR_EMBED_PROVIDER=azure
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scalablerag.llm.factory import make_embedder
from scalablerag.ingestion.corpus import CorpusIngester


def main():
    parser = argparse.ArgumentParser(description="Ingest a document corpus")
    parser.add_argument(
        "--source", required=True, help="Source directory containing documents"
    )
    parser.add_argument(
        "--store", required=True, help="Output directory for the indexed corpus"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="Minimum embeddings per document; auto-increased for large docs (default: 1)",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".txt"],
        help="File extensions to include (default: .txt)",
    )
    parser.add_argument(
        "--embed-provider",
        default=None,
        help="Embedding provider (huggingface, openai, ollama)",
    )
    parser.add_argument(
        "--embed-model", default=None, help="Embedding model name"
    )
    parser.add_argument(
        "--device", default=None, help="Device for HF embeddings (cpu, cuda)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Embedding batch size"
    )
    parser.add_argument(
        "--embeddings", action=argparse.BooleanOptionalAction, default=True,
    )

    args = parser.parse_args()

    skip_emb = not args.embeddings or os.environ.get("SR_NO_EMBEDDINGS") == "1"

    embedder = None
    if not skip_emb:
        print(f"Creating embedder (provider={args.embed_provider or 'auto'})...")
        embedder = make_embedder(
            provider=args.embed_provider,
            model=args.embed_model,
            device=args.device,
        )
        print(f"  Model dimension: {embedder.dimension}")
        print(f"  Max tokens: {embedder.max_tokens}")
    else:
        print("Embeddings DISABLED — docs only")

    ingester = CorpusIngester(
        store_dir=args.store, embedder=embedder, k=args.k
    )

    t0 = time.time()
    stats = ingester.ingest(
        source_dir=args.source,
        extensions=tuple(args.extensions),
        batch_size=args.batch_size,
    )
    elapsed = time.time() - t0

    print(f"\nIngestion complete in {elapsed:.1f}s")
    print(json.dumps(stats, indent=2, default=str))

    if stats.get("auto_promoted_count", 0) > 0:
        print(
            f"\n[INFO] {stats['auto_promoted_count']} documents needed more "
            f"than k={args.k} chunk(s) to avoid truncation. "
            f"Max chunks/doc: {stats['max_chunks_per_doc']}. "
            f"No information was lost."
        )


if __name__ == "__main__":
    main()
