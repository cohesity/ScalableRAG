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

"""Local GPU inference via a vLLM OpenAI-compatible server.

vLLM loads HuggingFace models (Llama, Mistral, Qwen, …) on GPU and serves
them through the *OpenAI Chat Completions HTTP API* — the same JSON
protocol OpenAI uses, but pointing at your machine instead of api.openai.com.

Set ``SR_LLM_PROVIDER=vllm`` and start a vLLM server locally, e.g.::

    python -m vllm.entrypoints.openai.api_server \\
        --model meta-llama/Llama-3.3-70B-Instruct \\
        --host 127.0.0.1 --port 8000

Then ScalableRAG connects to ``http://127.0.0.1:8000/v1`` by default.
"""

from __future__ import annotations

import os

from .base import LLMBudget
from .openai_provider import OpenAIProvider

_DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
_DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"


class VllmProvider(OpenAIProvider):
    """Client for a vLLM server exposing the OpenAI-compatible chat API."""

    def __init__(
        self,
        model: str | None = None,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        budget: LLMBudget | None = None,
    ) -> None:
        chosen_model = (
            model
            or os.environ.get("SR_VLLM_MODEL", "").strip()
            or os.environ.get("SR_LLM_MODEL", "").strip()
            or _DEFAULT_MODEL
        )
        chosen_base = (
            base_url
            or os.environ.get("SR_VLLM_BASE_URL", "").strip()
            or os.environ.get("SR_LLM_BASE_URL", "").strip()
            or _DEFAULT_BASE_URL
        )
        max_len = int(os.environ.get("SR_VLLM_MAX_MODEL_LEN", "32768"))
        os.environ.setdefault("SR_OPENAI_CONTEXT_LENGTH", str(max_len))
        os.environ.setdefault("SR_LLM_BASE_URL", chosen_base)

        key = (
            api_key
            or os.environ.get("SR_LLM_API_KEY", "").strip()
            or os.environ.get("OPENAI_API_KEY", "").strip()
            or "EMPTY"
        )

        super().__init__(
            model=chosen_model,
            api_key=key,
            azure_endpoint=None,
            api_version=None,
            budget=budget,
        )
        self._context_length = max_len
        self.max_completion_tokens = min(
            self.max_completion_tokens, self._context_length // 4
        )
