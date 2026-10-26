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

"""Corpus ingestion: load documents, embed, and persist.

Zero LLM calls at ingestion. ONLY embeddings.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ..llm.base import BaseEmbedder
from ..storage.doc_store import DocStore
from ..storage.embedding_store import EmbeddingStore
from ..storage.lexical_index import LexicalIndex


def _make_doc_id(filename: str) -> str:
    """Deterministic short doc ID from filename."""
    h = hashlib.sha256(filename.encode()).hexdigest()[:12]
    stem = Path(filename).stem[:60]
    safe = "".join(c if c.isalnum() or c in "_-." else "_" for c in stem)
    return f"{safe}__{h}"


def load_corpus_from_directory(
    source_dir: str | Path,
    extensions: tuple[str, ...] = (".txt",),
) -> list[tuple[str, str, str]]:
    """Scan a directory for documents.

    Returns list of (filepath, filename, text).
    """
    source = Path(source_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source}")

    docs: list[tuple[str, str, str]] = []
    for f in sorted(source.iterdir()):
        if not f.is_file():
            continue
        if extensions and f.suffix.lower() not in extensions:
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(f"[ingest] Skipping {f.name}: {e}")
            continue
        if text.strip():
            docs.append((str(f), f.name, text))

    return docs


def _chunk_text(text: str, chunk_size: int, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks by character count.

    Tries to break at paragraph or sentence boundaries.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            for sep in ["\n\n", "\n", ". ", " "]:
                look_back = text[max(start, end - 200) : end]
                idx = look_back.rfind(sep)
                if idx != -1:
                    end = max(start, end - 200) + idx + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap if end < len(text) else len(text)

    return chunks


class CorpusIngester:
    """Ingest documents from a directory, create embeddings, and persist.

    Losing information is never acceptable.  If a document does not fit in
    one embedding, the system automatically chunks it into as many pieces as
    needed so that every character is covered.

    Args:
        store_dir: Where to save the indexed corpus.
        embedder: Embedding model instance.
        k: *Minimum* number of embeddings per document.  Documents that are
           small enough will use exactly ``k`` chunks.  Documents that need
           more will get as many as required — the system never truncates.
    """

    def __init__(
        self,
        store_dir: str | Path,
        embedder: BaseEmbedder | None,
        k: int = 1,
        build_lexical_index: bool = False,
    ):
        self.store_dir = Path(store_dir)
        self.embedder = embedder
        self.k = k
        self.build_lexical_index = build_lexical_index
        self.doc_store = DocStore(self.store_dir / "docs")
        self.embedding_store = (
            EmbeddingStore(self.store_dir / "embeddings")
            if embedder is not None else None
        )

    def ingest(
        self,
        source_dir: str | Path,
        extensions: tuple[str, ...] = (".txt",),
        batch_size: int = 32,
    ) -> dict:
        """Load and embed all documents from source_dir.

        Returns a stats dict with counts.
        """
        raw_docs = load_corpus_from_directory(source_dir, extensions)
        print(f"[ingest] Found {len(raw_docs)} documents in {source_dir}")

        doc_ids = [_make_doc_id(filename) for _, filename, _ in raw_docs]
        self.doc_store.add_documents(
            (
                (doc_id, text, filename, {"source_path": filepath})
                for doc_id, (filepath, filename, text) in zip(doc_ids, raw_docs)
            )
        )

        lexical_index = None
        if self.build_lexical_index:
            print(f"[ingest] Building SQLite FTS5 lexical index for {len(doc_ids)} docs...")
            lexical_index = LexicalIndex.build(self.store_dir, self.doc_store)

        if self.embedder is None or self.embedding_store is None:
            stats = {
                "total_docs": len(doc_ids),
                "total_segments": 0,
                "embedding_dim": 0,
                "auto_promoted_count": 0,
                "total_chars": self.doc_store.total_chars(),
                "total_estimated_tokens": self.doc_store.total_estimated_tokens(),
                "lexical_index": str(lexical_index.path) if lexical_index else "",
            }
            print(f"[ingest] Done (docs only). {len(doc_ids)} docs")
            return stats

        embed_max_tokens = getattr(self.embedder, "max_tokens", 8192)
        embed_max_chars = int(embed_max_tokens * 1.5)

        texts_to_embed: list[str] = []
        chunks_per_doc: list[int] = []
        auto_promoted: list[str] = []

        for doc_id in doc_ids:
            text = self.doc_store.get_text(doc_id)

            if len(text) <= embed_max_chars:
                chunks = [text]
            else:
                chunks = _chunk_text(text, embed_max_chars, overlap=200)

            if len(chunks) > self.k:
                meta = self.doc_store.get_meta(doc_id)
                auto_promoted.append(
                    f"  {meta.filename}: {len(chunks)} chunks needed "
                    f"(requested k={self.k})"
                )

            while len(chunks) < self.k:
                chunks.append(text)

            for chunk in chunks:
                texts_to_embed.append(chunk)
            chunks_per_doc.append(len(chunks))

        if auto_promoted:
            print(
                f"\n[INFO] {len(auto_promoted)} document(s) required more "
                f"than k={self.k} chunk(s) to avoid truncation. "
                f"Auto-promoted:\n"
                + "\n".join(auto_promoted[:20])
            )
            if len(auto_promoted) > 20:
                print(f"  ... and {len(auto_promoted) - 20} more")
            print()

        print(
            f"[ingest] Embedding {len(texts_to_embed)} text segments "
            f"({len(doc_ids)} docs, requested k={self.k}, "
            f"max chunks/doc={max(chunks_per_doc)})..."
        )

        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(texts_to_embed), batch_size):
            batch = texts_to_embed[i : i + batch_size]
            emb = self.embedder.embed(batch)
            all_embeddings.append(emb)
            done = min(i + batch_size, len(texts_to_embed))
            print(
                f"  [{done}/{len(texts_to_embed)}] segments embedded",
                flush=True,
            )

        embeddings_array = np.vstack(all_embeddings)

        self.embedding_store.build(
            doc_ids,
            embeddings_array,
            chunks_per_doc=chunks_per_doc,
        )

        stats = {
            "total_docs": len(doc_ids),
            "total_segments": len(texts_to_embed),
            "requested_k": self.k,
            "max_chunks_per_doc": max(chunks_per_doc),
            "embedding_dim": embeddings_array.shape[1],
            "auto_promoted_count": len(auto_promoted),
            "auto_promoted_files": auto_promoted,
            "total_chars": self.doc_store.total_chars(),
            "total_estimated_tokens": self.doc_store.total_estimated_tokens(),
            "lexical_index": str(lexical_index.path) if lexical_index else "",
        }
        print(
            f"[ingest] Done. {stats['total_docs']} docs, "
            f"{stats['total_segments']} segments, "
            f"dim={stats['embedding_dim']}, "
            f"auto-promoted={stats['auto_promoted_count']}"
        )
        return stats
