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

"""Factory functions for creating LLM and Embedder instances."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.parse import urlparse

from .base import BaseLLM, BaseEmbedder, LLMBudget

_DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
_DEFAULT_VLLM_BASE_URL = "http://127.0.0.1:8000/v1"
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _ensure_ssl_certs() -> None:
    """Prefer OS trust store when certifi alone fails (e.g. corporate proxies)."""
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass


_ensure_ssl_certs()

KNOWN_LLM_PROVIDERS = ("openai", "azure", "gemini", "ollama", "vllm")
KNOWN_EMBED_PROVIDERS = ("openai", "azure", "ollama", "huggingface", "sentence-transformers")


def _warn_insecure_local_http(url: str, provider_name: str) -> None:
    """Warn when a local inference endpoint uses HTTP to a non-loopback host."""
    parsed = urlparse(url)
    if parsed.scheme != "http":
        return
    host = (parsed.hostname or "").lower()
    if not host or host in _LOOPBACK_HOSTS:
        return
    print(
        f"Warning: {provider_name} endpoint {url!r} uses unencrypted HTTP to a "
        f"non-loopback host ({host}). Prompts and responses may be intercepted. "
        f"Prefer loopback (127.0.0.1), TLS, or a trusted private network.",
        file=sys.stderr,
    )


def make_llm(
    *,
    provider: str | None = None,
    repo_root: Path | None = None,
    total_tokens: int = 500_000,
    model: str | None = None,
    azure_endpoint: str | None = None,
    azure_api_version: str | None = None,
    ollama_url: str | None = None,
    ollama_context_length: int | None = None,
    api_key: str | None = None,
) -> BaseLLM:
    """Create an LLM instance based on provider selection.

    Provider is resolved from: explicit arg > SR_LLM_PROVIDER env > 'openai'.
    API keys are resolved from: explicit arg > environment variables.
    """
    prov = (
        provider or os.environ.get("SR_LLM_PROVIDER", "openai")
    ).strip().lower()
    _ = repo_root

    if prov in {"openai", "azure"}:
        from .openai_provider import OpenAIProvider

        key = (
            api_key
            or os.environ.get("SR_LLM_API_KEY", "").strip()
            or os.environ.get("OPENAI_API_KEY", "").strip()
        )

        if prov == "azure":
            endpoint = azure_endpoint or os.environ.get("SR_AZURE_ENDPOINT", "").strip()
            api_ver = azure_api_version or os.environ.get(
                "SR_AZURE_API_VERSION", "2025-01-01-preview"
            )
        else:
            endpoint = azure_endpoint or ""
            api_ver = azure_api_version or ""
        chosen_model = model or os.environ.get("SR_LLM_MODEL", "gpt-4.1")

        return OpenAIProvider(
            model=chosen_model,
            api_key=key or None,
            azure_endpoint=endpoint or None,
            api_version=api_ver if endpoint else None,
            budget=LLMBudget(total_tokens=total_tokens),
        )

    if prov in {"gemini", "google"}:
        from .gemini_provider import GeminiProvider

        key = api_key or os.environ.get("GEMINI_KEY", "").strip()
        if not key:
            raise ValueError(
                "Gemini API key required: set the GEMINI_KEY environment variable."
            )
        chosen_model = model or os.environ.get("SR_GEMINI_MODEL", "gemini-2.5-flash")
        return GeminiProvider(
            model=chosen_model,
            api_key=key,
            budget=LLMBudget(total_tokens=total_tokens),
        )

    if prov == "ollama":
        from .local_provider import OllamaProvider

        chosen_model = model or os.environ.get("SR_OLLAMA_MODEL", "mistral")
        url = ollama_url or os.environ.get("SR_OLLAMA_URL", _DEFAULT_OLLAMA_URL)
        _warn_insecure_local_http(url, "Ollama")
        ctx = ollama_context_length or int(
            os.environ.get("SR_OLLAMA_CONTEXT_LENGTH", "8192")
        )
        return OllamaProvider(
            model=chosen_model, base_url=url, context_length=ctx
        )

    if prov == "vllm":
        from .vllm_provider import VllmProvider

        chosen_model = (
            model
            or os.environ.get("SR_VLLM_MODEL", "").strip()
            or os.environ.get("SR_LLM_MODEL", "").strip()
            or None
        )
        vllm_base = (
            os.environ.get("SR_VLLM_BASE_URL", "").strip()
            or os.environ.get("SR_LLM_BASE_URL", "").strip()
            or _DEFAULT_VLLM_BASE_URL
        )
        _warn_insecure_local_http(vllm_base, "vLLM")
        return VllmProvider(
            model=chosen_model,
            api_key=api_key,
            budget=LLMBudget(total_tokens=total_tokens),
        )

    raise ValueError(
        f"Unknown LLM provider: {prov!r}. "
        f"Expected one of: {', '.join(KNOWN_LLM_PROVIDERS)}."
    )


def make_embedder(
    *,
    provider: str | None = None,
    repo_root: Path | None = None,
    model: str | None = None,
    azure_endpoint: str | None = None,
    azure_api_version: str | None = None,
    ollama_url: str | None = None,
    device: str | None = None,
    api_key: str | None = None,
) -> BaseEmbedder:
    """Create an Embedder instance.

    Provider is resolved from: explicit arg > SR_EMBED_PROVIDER env > 'huggingface'.
    API keys are resolved from: explicit arg > environment variables.
    """
    prov = (
        provider or os.environ.get("SR_EMBED_PROVIDER", "huggingface")
    ).strip().lower()
    _ = repo_root

    if prov in {"openai", "azure"}:
        from .openai_provider import OpenAIEmbedder

        key = api_key or os.environ.get("OPENAI_API_KEY", "").strip()
        if prov == "azure":
            endpoint = azure_endpoint or os.environ.get("SR_AZURE_ENDPOINT", "").strip()
            api_ver = azure_api_version or os.environ.get(
                "SR_AZURE_API_VERSION", "2025-01-01-preview"
            )
        else:
            endpoint = azure_endpoint or ""
            api_ver = azure_api_version or ""
        chosen_model = model or os.environ.get(
            "SR_EMBED_MODEL", "text-embedding-3-small"
        )
        return OpenAIEmbedder(
            model=chosen_model,
            api_key=key or None,
            azure_endpoint=endpoint or None,
            api_version=api_ver if endpoint else None,
        )

    if prov in {"huggingface", "hf", "sentence-transformers"}:
        from .local_provider import SentenceTransformerEmbedder

        chosen_model = model or os.environ.get("SR_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        return SentenceTransformerEmbedder(model=chosen_model, device=device)

    if prov == "ollama":
        from .local_provider import OllamaEmbedder

        chosen_model = model or os.environ.get("SR_EMBED_MODEL", "nomic-embed-text")
        url = ollama_url or os.environ.get("SR_OLLAMA_URL", _DEFAULT_OLLAMA_URL)
        _warn_insecure_local_http(url, "Ollama")
        return OllamaEmbedder(model=chosen_model, base_url=url)

    raise ValueError(
        f"Unknown embedding provider: {prov!r}. "
        f"Expected one of: {', '.join(KNOWN_EMBED_PROVIDERS)}."
    )
