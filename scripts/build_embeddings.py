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

"""Build embeddings for an existing document index.

Reads docs from an existing index and creates a new embedding store
without re-ingesting the documents.  Useful for adding a second
embedding type (e.g. HuggingFace alongside existing OpenAI embeddings).

Usage:
    python scripts/build_embeddings.py \
        --store /path/to/index \
        --embed-provider huggingface \
        --embed-model sentence-transformers/all-MiniLM-L6-v2

The output directory is automatically named:
    {store}/embeddings__{provider}__{model-slug}/

Environment variables: for openai/azure providers, set OPENAI_API_KEY and SR_AZURE_ENDPOINT
(when provider=azure).
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scalablerag.llm.factory import make_embedder
from scalablerag.storage.doc_store import DocStore
from scalablerag.storage.embedding_store import EmbeddingStore


def _model_slug(model: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", model).strip("-").lower()
    return slug[:60]


def main():
    parser = argparse.ArgumentParser(
        description="Build embeddings for an existing index"
    )
    parser.add_argument("--store", required=True, help="Indexed corpus path")
    parser.add_argument("--embed-provider", default="huggingface")
    parser.add_argument("--embed-model", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--k", type=int, default=1,
        help="Embeddings per doc (auto-promoted for large docs)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Override output directory name (default: auto-named)",
    )

    args = parser.parse_args()
    store_path = Path(args.store)

    print("Loading doc store...")
    doc_store = DocStore(store_path / "docs")
    doc_ids = doc_store.all_doc_ids()
    print(f"  {len(doc_ids)} documents")

    print(f"Creating embedder ({args.embed_provider})...")
    embedder = make_embedder(
        provider=args.embed_provider,
        model=args.embed_model,
        device=args.device,
    )
    print(f"  dim={embedder.dimension}, max_tokens={embedder.max_tokens}")

    provider_slug = args.embed_provider or "hf"
    model_slug = _model_slug(args.embed_model or embedder.__class__.__name__)
    if args.output_dir:
        out_name = args.output_dir
    else:
        out_name = f"embeddings__{provider_slug}__{model_slug}"
    out_path = store_path / out_name
    print(f"Output: {out_path}")

    if out_path.exists():
        existing = EmbeddingStore(out_path)
        if existing.is_loaded and existing.doc_count == len(doc_ids):
            print(f"  Already exists with {existing.doc_count} docs — skipping")
            return
        print(f"  Exists but incomplete ({existing.doc_count} docs) — rebuilding")

    t0 = time.time()
    k = args.k
    batch_size = args.batch_size
    max_tokens = embedder.max_tokens

    all_embeddings = []
    chunks_per_doc = []

    for i in range(0, len(doc_ids), batch_size):
        batch_ids = doc_ids[i : i + batch_size]
        texts = []
        batch_chunks = []
        for did in batch_ids:
            text = doc_store.get_text(did)
            n_chunks = max(k, (len(text) // (max_tokens * 3)) + 1)
            if n_chunks == 1:
                texts.append(text[:max_tokens * 4])
                batch_chunks.append(1)
            else:
                chunk_size = len(text) // n_chunks
                for c in range(n_chunks):
                    start = c * chunk_size
                    end = start + chunk_size if c < n_chunks - 1 else len(text)
                    texts.append(text[start:end][:max_tokens * 4])
                batch_chunks.append(n_chunks)

        embs = embedder.embed(texts)
        all_embeddings.append(embs)
        chunks_per_doc.extend(batch_chunks)

        if (i // batch_size) % 10 == 0:
            done = min(i + batch_size, len(doc_ids))
            print(f"  Embedded {done}/{len(doc_ids)} docs...")

    all_emb = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    print(f"  Total vectors: {all_emb.shape}")

    embed_store = EmbeddingStore(out_path)
    embed_store.build(
        doc_ids=doc_ids,
        embeddings=all_emb,
        chunks_per_doc=chunks_per_doc,
        k=k,
        provider=provider_slug,
        model=args.embed_model or "",
    )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Store: {out_path}")
    print(f"  dim={embed_store.dimension}, docs={embed_store.doc_count}")


if __name__ == "__main__":
    main()
