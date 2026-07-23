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

"""Embedding store: persistent storage for document embeddings.

Uses FAISS (IndexFlatIP on L2-normalized vectors) for fast cosine similarity
search, with numpy memmap fallback when FAISS is unavailable.  Supports up
to 100k+ documents.
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path

try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False


class EmbeddingStore:
    """On-disk embedding store with optional FAISS acceleration.

    Layout::

        store_dir/
            embeddings.npy        # (N, D) float32 array
            embedding_index.json  # maps doc_id -> row indices
            config.json           # dimension, k, etc.
            faiss.index           # optional FAISS IndexFlatIP cache

    For k > 1 embeddings per doc, each doc occupies k consecutive rows.
    """

    def __init__(self, store_dir: str | Path):
        self.store_dir = Path(store_dir)
        self._embeddings: np.ndarray | None = None
        self._index: dict[str, list[int]] = {}
        self._config: dict = {}
        self._dimension: int = 0
        self._k: int = 1
        self._count: int = 0
        self._faiss_index = None
        self._provider: str = ""
        self._model: str = ""
        if self.store_dir.exists():
            self._load()

    def _load(self) -> None:
        cfg_path = self.store_dir / "config.json"
        if cfg_path.exists():
            self._config = json.loads(cfg_path.read_text())
            self._dimension = self._config.get("dimension", 0)
            self._k = self._config.get("k", 1)
            self._count = self._config.get("count", 0)
            self._provider = self._config.get("provider", "")
            self._model = self._config.get("model", "")

        idx_path = self.store_dir / "embedding_index.json"
        if idx_path.exists():
            self._index = json.loads(idx_path.read_text())

        emb_path = self.store_dir / "embeddings.npy"
        if emb_path.exists():
            self._embeddings = np.load(str(emb_path), mmap_mode="r")

        faiss_path = self.store_dir / "faiss.index"
        if _FAISS_AVAILABLE and faiss_path.exists():
            self._faiss_index = faiss.read_index(str(faiss_path))

    def _save_config(self) -> None:
        self.store_dir.mkdir(parents=True, exist_ok=True)
        cfg = {
            "dimension": self._dimension,
            "k": self._k,
            "count": self._count,
        }
        if self._provider:
            cfg["provider"] = self._provider
        if self._model:
            cfg["model"] = self._model
        (self.store_dir / "config.json").write_text(json.dumps(cfg))

    def _save_index(self) -> None:
        (self.store_dir / "embedding_index.json").write_text(
            json.dumps(self._index)
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def k(self) -> int:
        return self._k

    @property
    def doc_count(self) -> int:
        return len(self._index)

    @property
    def total_vectors(self) -> int:
        return self._count

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        return self._model

    @property
    def is_loaded(self) -> bool:
        """True if embeddings data exists and was loaded."""
        return self._embeddings is not None and self._dimension > 0

    @staticmethod
    def find_compatible_store(
        index_dir: Path,
        embedder_dim: int,
        embed_dir: str = "",
    ) -> "EmbeddingStore | None":
        """Find an embedding store in *index_dir* compatible with the embedder.

        Search order:
        1. If *embed_dir* is explicitly given, use it (error if dims mismatch).
        2. Scan ``embeddings__*`` subdirectories for a dimension match.
        3. Fall back to the legacy ``embeddings/`` directory if dims match.
        Returns None if nothing matches.
        """
        if embed_dir:
            path = index_dir / embed_dir
            if not path.exists():
                print(f"  [embed] Requested dir '{embed_dir}' not found in {index_dir}")
                return None
            store = EmbeddingStore(path)
            if store.is_loaded and store.dimension == embedder_dim:
                return store
            if store.is_loaded:
                print(
                    f"  [embed] Dimension mismatch: store={store.dimension}, "
                    f"embedder={embedder_dim} in {embed_dir}"
                )
            return None

        if not index_dir.exists():
            return None

        # Scan named dirs (embeddings__provider__model)
        for child in sorted(index_dir.iterdir()):
            if child.is_dir() and child.name.startswith("embeddings__"):
                store = EmbeddingStore(child)
                if store.is_loaded and store.dimension == embedder_dim:
                    print(f"  [embed] Auto-matched {child.name} (dim={embedder_dim})")
                    return store

        # Legacy fallback
        legacy = index_dir / "embeddings"
        if legacy.exists():
            store = EmbeddingStore(legacy)
            if store.is_loaded and store.dimension == embedder_dim:
                print(f"  [embed] Using legacy embeddings/ (dim={embedder_dim})")
                return store
            if store.is_loaded:
                print(
                    f"  [embed] Legacy embeddings/ dim={store.dimension} != "
                    f"embedder dim={embedder_dim} — skipping"
                )
        return None

    def build(
        self,
        doc_ids: list[str],
        embeddings: np.ndarray,
        chunks_per_doc: list[int] | None = None,
        k: int = 1,
        provider: str = "",
        model: str = "",
    ) -> None:
        """Build the store from scratch.

        Args:
            doc_ids: List of document IDs (length N).
            embeddings: (total_vectors, D) float32 array.
            chunks_per_doc: Number of chunks per doc (variable). If None,
                every doc is assumed to have exactly ``k`` chunks.
            k: Uniform chunks per doc (only used when chunks_per_doc is None).
        """
        if chunks_per_doc is not None:
            assert len(chunks_per_doc) == len(doc_ids)
            assert sum(chunks_per_doc) == embeddings.shape[0], (
                f"Expected {sum(chunks_per_doc)} vectors, "
                f"got {embeddings.shape[0]}"
            )
        else:
            n_docs = len(doc_ids)
            assert embeddings.shape[0] == n_docs * k, (
                f"Expected {n_docs * k} vectors, got {embeddings.shape[0]}"
            )
            chunks_per_doc = [k] * n_docs

        self._dimension = embeddings.shape[1]
        self._k = max(chunks_per_doc) if chunks_per_doc else 1
        self._count = embeddings.shape[0]
        self._provider = provider
        self._model = model

        self.store_dir.mkdir(parents=True, exist_ok=True)

        emb = embeddings.astype(np.float32)
        np.save(str(self.store_dir / "embeddings.npy"), emb)
        self._embeddings = np.load(
            str(self.store_dir / "embeddings.npy"), mmap_mode="r"
        )

        self._index = {}
        row = 0
        for i, doc_id in enumerate(doc_ids):
            n = chunks_per_doc[i]
            self._index[doc_id] = list(range(row, row + n))
            row += n

        self._build_faiss_index(emb)
        self._save_config()
        self._save_index()

    @staticmethod
    def _normalize_rows(mat: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms

    def _build_faiss_index(self, emb: np.ndarray) -> None:
        """Build a FAISS IndexFlatIP on L2-normalized vectors (cosine via IP)."""
        if not _FAISS_AVAILABLE:
            self._faiss_index = None
            return
        normed = self._normalize_rows(emb.copy())
        idx = faiss.IndexFlatIP(normed.shape[1])
        idx.add(normed)
        self._faiss_index = idx
        try:
            faiss.write_index(idx, str(self.store_dir / "faiss.index"))
        except Exception:
            pass

    def get_embedding(self, doc_id: str) -> np.ndarray:
        """Get embedding(s) for a document. Returns (k, D) array."""
        if doc_id not in self._index:
            raise KeyError(f"No embeddings for doc_id: {doc_id}")
        rows = self._index[doc_id]
        return np.array(self._embeddings[rows])

    def get_embeddings_for_ids(self, doc_ids: list[str]) -> np.ndarray:
        """Get mean embeddings for a list of docs. Returns (len(doc_ids), D)."""
        vecs = []
        for did in doc_ids:
            emb = self.get_embedding(did)
            vecs.append(emb.mean(axis=0) if emb.ndim == 2 else emb)
        return np.stack(vecs)

    def get_all_mean_embeddings(self) -> tuple[list[str], np.ndarray]:
        """Get mean embedding per doc for all docs. Returns (doc_ids, (N, D))."""
        doc_ids = list(self._index.keys())
        return doc_ids, self.get_embeddings_for_ids(doc_ids)

    def cosine_search(
        self,
        query_vec: np.ndarray,
        top_k: int = 10,
        candidate_ids: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Find top-k most similar documents by cosine similarity.

        Uses FAISS IndexFlatIP when available (fast); falls back to numpy.

        Args:
            query_vec: (D,) query embedding.
            top_k: Number of results.
            candidate_ids: Restrict search to these doc IDs (None = all).

        Returns:
            List of (doc_id, similarity_score) sorted descending.
        """
        if candidate_ids is not None:
            search_ids = [d for d in candidate_ids if d in self._index]
        else:
            search_ids = list(self._index.keys())

        if not search_ids:
            return []

        qv = query_vec.astype(np.float32).reshape(1, -1)
        qv = self._normalize_rows(qv)

        # Fast path: FAISS over all docs when no candidate filter
        if self._faiss_index is not None and candidate_ids is None:
            sims, idxs = self._faiss_index.search(qv, top_k)
            all_doc_ids = list(self._index.keys())
            row_to_doc: dict[int, str] = {}
            for did, rows in self._index.items():
                for r in rows:
                    row_to_doc[r] = did

            seen: set[str] = set()
            results: list[tuple[str, float]] = []
            for sim, idx in zip(sims[0].tolist(), idxs[0].tolist()):
                if idx < 0:
                    continue
                did = row_to_doc.get(idx)
                if did and did not in seen:
                    seen.add(did)
                    results.append((did, float(sim)))
            return results[:top_k]

        # Filtered or no-FAISS path: brute-force numpy cosine
        doc_vecs = self.get_embeddings_for_ids(search_ids)
        doc_vecs_normed = self._normalize_rows(doc_vecs.astype(np.float32))
        sims_arr = (doc_vecs_normed @ qv[0])

        top_idx = np.argsort(sims_arr)[::-1][:top_k]
        return [(search_ids[i], float(sims_arr[i])) for i in top_idx]

    def all_doc_ids(self) -> list[str]:
        return list(self._index.keys())
