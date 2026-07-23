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

"""Run the retrieval agent to answer a question over an indexed corpus.

Usage:
    python scripts/query.py --store /path/to/store --question "What is ...?"
    python scripts/query.py --regime zero --store INDEX --question "..."

Environment variables:
    SR_LLM_PROVIDER     openai | azure | gemini | ollama | vllm (default: openai)
    SR_LLM_MODEL        Model or Azure deployment name (default: gpt-4.1)
    OPENAI_API_KEY      Required for openai/azure LLM
    SR_LLM_API_KEY      Optional alias for OPENAI_API_KEY
    GEMINI_KEY          Required when SR_LLM_PROVIDER=gemini
    SR_AZURE_ENDPOINT   Required when SR_LLM_PROVIDER=azure
    SR_VLLM_MODEL       HuggingFace id when provider=vllm
    SR_EMBED_PROVIDER   Must match what was used at ingestion
    SR_EMBED_MODEL      Must match what was used at ingestion
"""

from __future__ import annotations

import argparse
import sys
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


def main():
    parser = argparse.ArgumentParser(
        description="Query an indexed corpus using the retrieval agent"
    )
    parser.add_argument(
        "--store", required=True, help="Path to indexed corpus directory"
    )
    parser.add_argument(
        "--question", "-q", required=True, help="Question to answer"
    )
    parser.add_argument(
        "--regime",
        choices=["zero", "limited"],
        default=None,
        help="zero or limited",
    )
    parser.add_argument(
        "--llm-provider", default=None, help="LLM provider"
    )
    parser.add_argument("--llm-model", default=None, help="LLM model name")
    parser.add_argument(
        "--embed-provider", default=None, help="Embedding provider"
    )
    parser.add_argument(
        "--embed-model", default=None, help="Embedding model name"
    )
    parser.add_argument(
        "--embed-dir", default=None,
        help="Embedding subdirectory name (e.g. embeddings__openai__text-embedding-3-small)",
    )
    parser.add_argument(
        "--embeddings", action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--agg-tools", action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--branch-select", action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--max-steps", type=int, default=25, help="Max agent steps"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress step-by-step output"
    )

    args = parser.parse_args()
    cfg = resolve_run_config(args, detect_explicit_flags(sys.argv))

    store_path = Path(args.store)
    if not store_path.exists():
        print(f"Error: store not found at {store_path}")
        sys.exit(1)

    print("Loading corpus...")
    doc_store = DocStore(store_path / "docs")

    embedding_store = None
    embedder = None
    use_embeddings = False
    if cfg.skip_embeddings:
        print("  Embeddings DISABLED")
    else:
        embedding_store, embedder, use_embeddings = load_embedding_setup(
            store_path,
            skip_embeddings=False,
            embed_provider=cfg.embed_provider,
            embed_model=cfg.embed_model,
            embed_dir=cfg.embed_dir,
        )
        if use_embeddings and embedding_store is not None:
            print(
                f"  {doc_store.doc_count()} docs, "
                f"{embedding_store.total_vectors} vectors, "
                f"dim={embedding_store.dimension}"
            )
        elif cfg.require_embeddings:
            print("ERROR: no compatible embedding store")
            sys.exit(1)
        else:
            print(f"  {doc_store.doc_count()} docs, no embedding store")

    print("Creating LLM...")
    llm = make_llm(provider=args.llm_provider, model=args.llm_model)
    print(f"  Context length: {llm.context_length}")

    agent = RetrievalAgent(
        llm=llm,
        doc_store=doc_store,
        embedding_store=embedding_store,
        embedder=embedder,
        max_steps=args.max_steps,
        use_embeddings=use_embeddings,
        branch_select=cfg.branch_select,
        use_agg_tools=cfg.use_agg_tools,
    )

    print(f"\nQuestion: {args.question}\n")
    result = agent.query(args.question, verbose=not args.quiet)

    print("\n" + "=" * 60)
    print("ANSWER:", result["answer"])
    print("=" * 60)
    print(
        f"Steps: {result['steps']} | "
        f"Tool calls: {len(result['tool_calls'])} | "
        f"Time: {result['elapsed_s']}s"
    )
    if result.get("forced"):
        print("(Answer was forced after reaching max steps)")


if __name__ == "__main__":
    main()
