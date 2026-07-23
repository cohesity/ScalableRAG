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

"""OpenAI / Azure OpenAI LLM and Embedder implementations."""

from __future__ import annotations

import os
import random
import re
import time
from typing import Any

import numpy as np

from .base import BaseLLM, BaseEmbedder, LLMBudget

try:
    import openai

    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False


def _extract_retry_after(err: Exception) -> int | None:
    m = re.search(r"retry after\s+(\d+)\s+seconds", str(err), flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def _is_transient(err: Exception) -> bool:
    transient_names = {
        "RateLimitError",
        "APITimeoutError",
        "APIConnectionError",
        "InternalServerError",
        "ServiceUnavailableError",
        "APIError",
    }
    if err.__class__.__name__ in transient_names:
        return True
    msg = str(err).lower()
    return any(
        s in msg
        for s in [
            "rate limit",
            "timeout",
            "timed out",
            "temporarily unavailable",
            "service unavailable",
            "connection reset",
            "connection error",
            "server error",
            "502",
            "503",
            "504",
        ]
    )


class ContentFilterError(Exception):
    """Azure content management policy blocked the request."""


def _is_content_filter(e: Exception) -> bool:
    if e.__class__.__name__ == "BadRequestError":
        msg = str(e).lower()
        if "content management policy" in msg or "content_filter" in msg:
            return True
    return False


def _is_context_length_error(e: Exception) -> bool:
    if e.__class__.__name__ == "BadRequestError":
        msg = str(e).lower()
        if "maximum context length" in msg or "too many tokens" in msg:
            return True
    return False


def _call_with_retry(op_name: str, fn, *, model: str | None = None) -> Any:
    """Call *fn()* with infinite retry + exponential backoff.

    Non-retryable errors (e.g. content filter) are raised immediately.
    """
    attempt = 0
    t0 = time.monotonic()
    last_sleep = 0.0

    while True:
        try:
            return fn()
        except Exception as e:
            if _is_content_filter(e):
                print(
                    f"[OpenAI] {op_name} BLOCKED by content filter (non-retryable). "
                    f"model={model} err={str(e).replace(chr(10), ' ')[:200]}",
                    flush=True,
                )
                raise ContentFilterError(str(e)) from e

            if _is_context_length_error(e):
                print(
                    f"[OpenAI] {op_name} context length exceeded (non-retryable). "
                    f"model={model} err={str(e).replace(chr(10), ' ')[:200]}",
                    flush=True,
                )
                raise

            attempt += 1
            elapsed = time.monotonic() - t0
            retry_after = _extract_retry_after(e)

            if _is_transient(e):
                base = min(30.0, min(120.0, 2.0 ** min(12, attempt)))
            else:
                base = min(300.0, 5.0 * max(1.0, attempt / 5.0))

            sleep_s = float(max(base, float(retry_after or 0))) + random.uniform(0, 1)

            should_print = (
                attempt <= 5
                or attempt % 5 == 0
                or sleep_s >= max(30.0, last_sleep + 10.0)
            )
            if should_print:
                kind = e.__class__.__name__
                msg = str(e).replace("\n", " ")[:240]
                print(
                    f"[OpenAI] {op_name} failed ({kind}). attempt={attempt} "
                    f"elapsed={elapsed:.1f}s sleep={sleep_s:.1f}s model={model} "
                    f"err={msg}",
                    flush=True,
                )
            last_sleep = sleep_s
            time.sleep(sleep_s)


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------


class OpenAIProvider(BaseLLM):
    """OpenAI (or Azure OpenAI) chat completion provider."""

    def __init__(
        self,
        model: str = "gpt-4.1",
        api_key: str | None = None,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
        budget: LLMBudget | None = None,
    ):
        super().__init__(budget)
        if not _OPENAI_AVAILABLE:
            raise ImportError("openai package not installed. Run: pip install openai")

        self.model = model
        self.azure_endpoint = azure_endpoint
        self.temperature = float(os.environ.get("SR_LLM_TEMPERATURE", "0.0"))

        ml = model.lower()
        if "gpt-4.1" in ml:
            self._context_length = int(
                os.environ.get("SR_OPENAI_CONTEXT_LENGTH", "1000000")
            )
        elif "gpt-4o" in ml or "o4" in ml or "o3" in ml:
            self._context_length = int(
                os.environ.get("SR_OPENAI_CONTEXT_LENGTH", "128000")
            )
        else:
            self._context_length = int(
                os.environ.get("SR_OPENAI_CONTEXT_LENGTH", "128000")
            )
        self.max_completion_tokens = min(
            self.max_completion_tokens, self._context_length // 4,
        )

        if azure_endpoint:
            if not api_version:
                raise ValueError("api_version required for Azure endpoint")
            self.client = openai.AzureOpenAI(
                api_key=api_key,
                azure_endpoint=azure_endpoint,
                api_version=api_version,
                max_retries=0,
            )
        else:
            kw: dict[str, Any] = {"max_retries": 0}
            if api_key:
                kw["api_key"] = api_key
            base_url = os.environ.get("SR_LLM_BASE_URL")
            if base_url:
                kw["base_url"] = base_url
            self.client = openai.OpenAI(**kw)

    # -- single prompt ---------------------------------------------------

    def complete(
        self,
        prompt: str,
        max_tokens: int = 1024,
        *,
        json_schema: dict | None = None,
    ) -> str:
        max_tokens = min(max_tokens, self.max_completion_tokens)
        self._maybe_expand_budget(prompt, max_tokens)

        create_kw: dict[str, Any] = dict(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=self.temperature,
            timeout=600.0,
        )
        if json_schema is not None:
            create_kw["response_format"] = {"type": "json_object"}

        resp = _call_with_retry(
            "chat.completions.create",
            lambda: self.client.chat.completions.create(**create_kw),
            model=self.model,
        )
        usage = resp.usage
        if usage:
            self._record_usage(usage.total_tokens)
        if getattr(resp.choices[0], "finish_reason", None) == "length":
            ct = usage.completion_tokens if usage else 0
            self._flag_output_truncation(ct, max_tokens, "OpenAI")

        return resp.choices[0].message.content or ""

    # -- multi-turn chat -------------------------------------------------

    def complete_chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        *,
        json_schema: dict | None = None,
    ) -> str:
        max_tokens = min(max_tokens, self.max_completion_tokens)
        total_chars = sum(len(m.get("content", "")) for m in messages)
        self._maybe_expand_budget_raw(total_chars, max_tokens)

        api_msgs = [
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in messages
        ]
        create_kw: dict[str, Any] = dict(
            model=self.model,
            messages=api_msgs,
            max_tokens=max_tokens,
            temperature=self.temperature,
            timeout=600.0,
        )
        if json_schema is not None:
            create_kw["response_format"] = {"type": "json_object"}

        resp = _call_with_retry(
            "chat.completions.create",
            lambda: self.client.chat.completions.create(**create_kw),
            model=self.model,
        )
        usage = resp.usage
        if usage:
            self._record_usage(usage.total_tokens)
        if getattr(resp.choices[0], "finish_reason", None) == "length":
            ct = usage.completion_tokens if usage else 0
            self._flag_output_truncation(ct, max_tokens, "OpenAI/chat")

        return resp.choices[0].message.content or ""

    # -- budget helpers --------------------------------------------------

    def _maybe_expand_budget(self, prompt: str, max_tokens: int) -> None:
        self._maybe_expand_budget_raw(len(prompt), max_tokens)

    def _maybe_expand_budget_raw(self, input_chars: int, max_tokens: int) -> None:
        estimated = input_chars // 4 + max_tokens
        if self.can_afford(estimated):
            return
        if self._budget is None:
            return
        new = self._budget.used_tokens + estimated + 5_000_000
        self._budget.total_tokens = max(self._budget.total_tokens, new)
        print(
            f"[LLM budget] Auto-expanding: new_total={self._budget.total_tokens}",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI (or Azure) text embedding."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
    ):
        if not _OPENAI_AVAILABLE:
            raise ImportError("openai package not installed. Run: pip install openai")

        self.model = model
        self._max_tokens = 8191

        if azure_endpoint:
            if not api_version:
                raise ValueError("api_version required for Azure endpoint")
            self.client = openai.AzureOpenAI(
                api_key=api_key,
                azure_endpoint=azure_endpoint,
                api_version=api_version,
                max_retries=0,
            )
        else:
            kw: dict[str, Any] = {"max_retries": 0}
            if api_key:
                kw["api_key"] = api_key
            self.client = openai.OpenAI(**kw)

        self._dimension = 1536 if "small" in model else 3072

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    def _embed_single_safe(self, text: str) -> list[float]:
        """Embed a single text, splitting if it exceeds the token limit."""
        try:
            resp = _call_with_retry(
                "embeddings.create",
                lambda: self.client.embeddings.create(
                    model=self.model, input=[text]
                ),
                model=self.model,
            )
            return resp.data[0].embedding
        except Exception as e:
            if _is_context_length_error(e):
                mid = len(text) // 2
                for sep in ["\n\n", "\n", ". ", " "]:
                    idx = text.rfind(sep, mid - 500, mid + 500)
                    if idx != -1:
                        mid = idx + len(sep)
                        break
                left = text[:mid].strip()
                right = text[mid:].strip()
                print(
                    f"[OpenAI] Text too long ({len(text)} chars), "
                    f"splitting into {len(left)}+{len(right)} chars",
                    flush=True,
                )
                parts = []
                if left:
                    parts.append(np.array(self._embed_single_safe(left)))
                if right:
                    parts.append(np.array(self._embed_single_safe(right)))
                if not parts:
                    return [0.0] * self._dimension
                avg = np.mean(parts, axis=0)
                avg = avg / (np.linalg.norm(avg) + 1e-12)
                return avg.tolist()
            raise

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        all_emb: list[list[float]] = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                resp = _call_with_retry(
                    "embeddings.create",
                    lambda b=batch: self.client.embeddings.create(
                        model=self.model, input=b
                    ),
                    model=self.model,
                )
                all_emb.extend(item.embedding for item in resp.data)
            except Exception as e:
                if _is_context_length_error(e):
                    print(
                        f"[OpenAI] Batch too long, falling back to per-text embedding",
                        flush=True,
                    )
                    for text in batch:
                        all_emb.append(self._embed_single_safe(text))
                else:
                    raise
        return np.array(all_emb, dtype=np.float32)

    def embed_single(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        return self._dimension
