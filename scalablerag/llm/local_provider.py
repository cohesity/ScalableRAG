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

"""Ollama LLM and Embedder implementations for local inference."""

from __future__ import annotations

import numpy as np
import httpx

from .base import BaseLLM, BaseEmbedder, LLMBudget


class OllamaProvider(BaseLLM):
    """Local LLM via Ollama HTTP API."""

    def __init__(
        self,
        model: str = "mistral",
        base_url: str = "http://127.0.0.1:11434",
        budget: LLMBudget | None = None,
        timeout: float = 120.0,
        context_length: int | None = None,
    ):
        super().__init__(budget)
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)
        if context_length:
            self._context_length = context_length
        else:
            self._context_length = 8192

    def complete(
        self,
        prompt: str,
        max_tokens: int = 1024,
        *,
        json_schema: dict | None = None,
    ) -> str:
        max_tokens = min(max_tokens, self.max_completion_tokens)
        est = len(prompt) // 4 + max_tokens
        if not self.can_afford(est):
            raise RuntimeError(
                "LLM budget exhausted. Increase LLMBudget.total_tokens."
            )

        options: dict = {
            "num_predict": max_tokens,
            "temperature": 0.1,
        }
        body: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if json_schema is not None:
            body["format"] = "json"

        try:
            r = self._client.post(f"{self.base_url}/api/generate", json=body)
            r.raise_for_status()
            data = r.json()
        except httpx.HTTPError as e:
            raise RuntimeError(f"Ollama request failed: {e}") from e

        out_toks = data.get("eval_count", max_tokens // 2)
        in_toks = data.get("prompt_eval_count", len(prompt) // 4)
        self._record_usage(in_toks + out_toks)

        return data.get("response", "")

    def complete_chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        *,
        json_schema: dict | None = None,
    ) -> str:
        """Multi-turn chat via Ollama /api/chat."""
        max_tokens = min(max_tokens, self.max_completion_tokens)
        total_chars = sum(len(m.get("content", "")) for m in messages)
        est = total_chars // 4 + max_tokens
        if not self.can_afford(est):
            raise RuntimeError("LLM budget exhausted.")

        body: dict = {
            "model": self.model,
            "messages": [
                {"role": m.get("role", "user"), "content": m.get("content", "")}
                for m in messages
            ],
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.1,
            },
        }
        if json_schema is not None:
            body["format"] = "json"

        try:
            r = self._client.post(f"{self.base_url}/api/chat", json=body)
            r.raise_for_status()
            data = r.json()
        except httpx.HTTPError as e:
            raise RuntimeError(f"Ollama chat request failed: {e}") from e

        out_toks = data.get("eval_count", max_tokens // 2)
        in_toks = data.get("prompt_eval_count", total_chars // 4)
        self._record_usage(in_toks + out_toks)

        msg = data.get("message", {})
        return msg.get("content", "")

    def __del__(self):
        if hasattr(self, "_client"):
            try:
                self._client.close()
            except Exception:
                pass


class OllamaEmbedder(BaseEmbedder):
    """Local embeddings via Ollama."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 60.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)
        self._dimension: int | None = None
        self._max_tokens = 8192

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            dim = self._dimension or 768
            return np.zeros((0, dim), dtype=np.float32)
        vecs = [self.embed_single(t) for t in texts]
        return np.stack(vecs)

    def embed_single(self, text: str) -> np.ndarray:
        try:
            r = self._client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
            )
            r.raise_for_status()
            emb = r.json().get("embedding", [])
            if self._dimension is None and emb:
                self._dimension = len(emb)
            return np.array(emb, dtype=np.float32)
        except httpx.HTTPError as e:
            raise RuntimeError(f"Ollama embedding failed: {e}") from e

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self.embed_single("dimension probe")
        return self._dimension or 768

    def __del__(self):
        if hasattr(self, "_client"):
            try:
                self._client.close()
            except Exception:
                pass


class SentenceTransformerEmbedder(BaseEmbedder):
    """Local embeddings via sentence-transformers (HuggingFace models)."""

    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
        batch_size: int = 64,
        local_files_only: bool = False,
    ):
        import os

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers required. Install: pip install sentence-transformers"
            ) from e

        if local_files_only:
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            os.environ.setdefault("HF_HUB_OFFLINE", "1")

        self._model = SentenceTransformer(
            model, device=device, local_files_only=local_files_only
        )
        self._batch_size = batch_size
        self._dim: int | None = None
        self._max_tokens = self._model.max_seq_length or 8192

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            dim = self._dim or 1024
            return np.zeros((0, dim), dtype=np.float32)
        vecs = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,
        )
        arr = np.array(vecs, dtype=np.float32)
        if self._dim is None and arr.ndim == 2:
            self._dim = arr.shape[1]
        return arr

    def embed_single(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        if self._dim is None:
            self.embed_single("dimension probe")
        return self._dim or 1024
