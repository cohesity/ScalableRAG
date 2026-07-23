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

"""Gemini LLM provider (Google models via Vertex AI public endpoint + API key)."""

from __future__ import annotations

import json
import os
import random
import time
from typing import Any

import httpx

from .base import BaseLLM, LLMBudget


def _is_transient_http(status: int) -> bool:
    return status in {408, 429, 500, 502, 503, 504}


def _call_with_retry(op_name: str, fn) -> Any:
    attempt = 0
    t0 = time.monotonic()
    last_sleep = 0.0
    while True:
        try:
            return fn()
        except Exception as e:
            attempt += 1
            elapsed = time.monotonic() - t0
            base = min(30.0, min(120.0, 2.0 ** min(12, attempt)))
            sleep_s = base + random.uniform(0.0, 1.0)
            should_print = (
                attempt <= 5
                or attempt % 5 == 0
                or sleep_s >= max(30.0, last_sleep + 10.0)
            )
            if should_print:
                msg = str(e).replace("\n", " ")[:240]
                print(
                    f"[Gemini] {op_name} failed ({e.__class__.__name__}). "
                    f"attempt={attempt} elapsed={elapsed:.1f}s sleep={sleep_s:.1f}s "
                    f"err={msg}",
                    flush=True,
                )
            last_sleep = sleep_s
            time.sleep(sleep_s)


def _extract_text(obj: Any) -> str:
    """Extract concatenated text from streamGenerateContent output."""
    if obj is None:
        return ""
    chunks = obj if isinstance(obj, list) else [obj]
    parts: list[str] = []
    for ch in chunks:
        if not isinstance(ch, dict):
            continue
        for cand in ch.get("candidates") or []:
            if not isinstance(cand, dict):
                continue
            content = cand.get("content")
            if not isinstance(content, dict):
                continue
            for p in content.get("parts") or []:
                if isinstance(p, dict):
                    t = p.get("text")
                    if isinstance(t, str) and t:
                        parts.append(t)
    return "".join(parts).strip()


def _extract_usage_finish(obj: Any) -> tuple[dict | None, str | None]:
    last = None
    if isinstance(obj, list) and obj:
        last = obj[-1]
    elif isinstance(obj, dict):
        last = obj
    if not isinstance(last, dict):
        return None, None
    usage = last.get("usageMetadata")
    if not isinstance(usage, dict):
        usage = None
    finish = None
    cands = last.get("candidates")
    if isinstance(cands, list) and cands:
        c0 = cands[0]
        if isinstance(c0, dict):
            fr = c0.get("finishReason")
            if isinstance(fr, str):
                finish = fr
    return usage, finish


class GeminiProvider(BaseLLM):
    """Gemini LLM via public Vertex AI endpoint + API key."""

    def __init__(
        self,
        *,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
        budget: LLMBudget | None = None,
        timeout_s: float = 120.0,
        base_url: str = "https://aiplatform.googleapis.com/v1/publishers/google/models",
    ):
        super().__init__(budget)
        self.model = model
        self.api_key = (api_key or "").strip()
        if not self.api_key:
            raise ValueError("Gemini API key is required.")
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout_s)
        self._context_length = 1_000_000

    def complete(
        self,
        prompt: str,
        max_tokens: int = 1024,
        *,
        json_schema: dict | None = None,
    ) -> str:
        max_tokens = min(max_tokens, self.max_completion_tokens)
        est = len(prompt) // 4 + max_tokens
        if not self.can_afford(est) and self._budget is not None:
            new = self._budget.used_tokens + est + 5_000_000
            self._budget.total_tokens = max(self._budget.total_tokens, new)

        url = f"{self.base_url}/{self.model}:streamGenerateContent"
        params = {"key": self.api_key}
        wants_json = json_schema is not None
        gen_cfg: dict[str, Any] = {
            "temperature": 0.1,
            "maxOutputTokens": max_tokens,
        }
        if wants_json:
            gen_cfg["responseMimeType"] = "application/json"

        tl = os.environ.get("SR_GEMINI_THINKING_LEVEL", "").strip().upper()
        tb = os.environ.get("SR_GEMINI_THINKING_BUDGET", "").strip()
        if tl or tb:
            tc: dict[str, Any] = {}
            if tl in {"LOW", "HIGH"}:
                tc["thinkingLevel"] = tl
            if tb:
                try:
                    tc["thinkingBudget"] = int(tb)
                except ValueError:
                    pass
            if tc:
                gen_cfg["thinkingConfig"] = tc

        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": gen_cfg,
        }

        def _do():
            r = self._client.post(
                url,
                params=params,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            if _is_transient_http(r.status_code):
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
            r.raise_for_status()
            return r

        r = _call_with_retry("streamGenerateContent", _do)
        text = ""
        usage = None
        finish = None
        try:
            obj = r.json()
            text = _extract_text(obj)
            usage, finish = _extract_usage_finish(obj)
        except Exception:
            raw = (r.text or "").strip()
            if raw:
                parts: list[str] = []
                last_obj = None
                for line in raw.splitlines():
                    line = line.strip()
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        last_obj = obj
                        t = _extract_text(obj)
                        if t:
                            parts.append(t)
                    except Exception:
                        continue
                text = "".join(parts).strip()
                usage, finish = _extract_usage_finish(last_obj)

        if text:
            self._record_usage(len(prompt) // 4 + len(text) // 4)
        else:
            self._record_usage(max(1, len(prompt) // 4))

        if finish and "MAX_TOKENS" in str(finish).upper():
            ct = (usage or {}).get("candidatesTokenCount", len(text) // 4)
            self._flag_output_truncation(ct, max_tokens, "Gemini")

        return text

    def complete_chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        *,
        json_schema: dict | None = None,
    ) -> str:
        """Multi-turn via Gemini by converting to contents list."""
        max_tokens = min(max_tokens, self.max_completion_tokens)

        contents: list[dict[str, Any]] = []
        system_parts: list[str] = []
        for m in messages:
            role = m.get("role", "user")
            text = m.get("content", "")
            if role == "system":
                system_parts.append(text)
            else:
                gemini_role = "model" if role == "assistant" else "user"
                contents.append(
                    {"role": gemini_role, "parts": [{"text": text}]}
                )

        if system_parts and contents:
            prefix = "\n".join(system_parts) + "\n\n"
            contents[0]["parts"][0]["text"] = (
                prefix + contents[0]["parts"][0]["text"]
            )

        total_chars = sum(len(m.get("content", "")) for m in messages)
        est = total_chars // 4 + max_tokens
        if not self.can_afford(est) and self._budget is not None:
            new = self._budget.used_tokens + est + 5_000_000
            self._budget.total_tokens = max(self._budget.total_tokens, new)

        url = f"{self.base_url}/{self.model}:streamGenerateContent"
        params = {"key": self.api_key}
        gen_cfg: dict[str, Any] = {
            "temperature": 0.1,
            "maxOutputTokens": max_tokens,
        }
        if json_schema is not None:
            gen_cfg["responseMimeType"] = "application/json"

        payload = {"contents": contents, "generationConfig": gen_cfg}

        def _do():
            r = self._client.post(
                url,
                params=params,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            if _is_transient_http(r.status_code):
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
            r.raise_for_status()
            return r

        r = _call_with_retry("streamGenerateContent/chat", _do)
        text = ""
        try:
            obj = r.json()
            text = _extract_text(obj)
        except Exception:
            pass

        if text:
            self._record_usage(total_chars // 4 + len(text) // 4)
        return text

    def __del__(self):
        try:
            if hasattr(self, "_client"):
                self._client.close()
        except Exception:
            pass
