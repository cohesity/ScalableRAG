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

"""Abstract base classes for LLM and Embedder providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class LLMBudget:
    """Token budget tracking for LLM usage."""

    total_tokens: int
    used_tokens: int = 0

    def remaining(self) -> int:
        return self.total_tokens - self.used_tokens

    def can_afford(self, estimated_tokens: int) -> bool:
        return self.remaining() >= estimated_tokens

    def debit(self, tokens: int) -> None:
        self.used_tokens += tokens

    def exhausted(self) -> bool:
        return self.used_tokens >= self.total_tokens


class BaseLLM(ABC):
    """Abstract base class for LLM providers.

    Budget is optional.  When *budget* is ``None`` (e.g. local GPU inference),
    ``can_afford()`` always returns True and no usage is tracked.
    """

    max_completion_tokens: int = 32768

    def __init__(self, budget: LLMBudget | None = None):
        self._budget = budget
        if self._budget is not None and not isinstance(self._budget, LLMBudget):
            self._budget = LLMBudget(total_tokens=100_000)
        self._output_truncations = 0

    @abstractmethod
    def complete(
        self,
        prompt: str,
        max_tokens: int = 1024,
        *,
        json_schema: dict | None = None,
    ) -> str:
        """Generate a completion for the given prompt."""
        ...

    def complete_batch(
        self,
        prompts: list[str],
        max_tokens: int = 1024,
        *,
        json_schema: dict | None = None,
    ) -> list[str]:
        """Generate completions for multiple prompts (sequential by default)."""
        return [
            self.complete(p, max_tokens=max_tokens, json_schema=json_schema)
            for p in prompts
        ]

    def complete_chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        *,
        json_schema: dict | None = None,
    ) -> str:
        """Multi-turn chat completion.

        Default implementation flattens to a single prompt.
        """
        parts: list[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                parts.append(content)
            else:
                parts.append(f"{role}: {content}")
        return self.complete(
            "\n\n".join(parts),
            max_tokens=max_tokens,
            json_schema=json_schema,
        )

    @property
    def context_length(self) -> int:
        """Max context window in tokens."""
        return getattr(self, "_context_length", 32768)

    def count_tokens(self, text: str) -> int:
        """Estimate the number of tokens in *text* (~3.5 chars/token)."""
        return max(1, len(text) * 10 // 35)

    def prompt_fits(
        self, prompt: str, max_tokens: int
    ) -> tuple[bool, int, int]:
        """Check whether *prompt* fits with room for *max_tokens* completion.

        Returns ``(fits, prompt_tokens, available_input_tokens)``.
        """
        prompt_tokens = self.count_tokens(prompt)
        available = max(0, self.context_length - max_tokens)
        return (prompt_tokens <= available, prompt_tokens, available)

    def budget_remaining(self) -> int:
        if self._budget is None:
            return 2**63
        return self._budget.remaining()

    def can_afford(self, estimated_tokens: int) -> bool:
        if self._budget is None:
            return True
        return self._budget.can_afford(estimated_tokens)

    def _record_usage(self, tokens: int) -> None:
        if self._budget is not None:
            self._budget.debit(tokens)

    @property
    def output_truncations(self) -> int:
        return self._output_truncations

    def _flag_output_truncation(
        self, completion_tokens: int, max_tokens: int, provider_tag: str
    ) -> None:
        self._output_truncations += 1
        print(
            f"[OUTPUT-TRUNCATED] ({provider_tag}) Generation hit the "
            f"max_tokens cap ({completion_tokens}/{max_tokens} tokens). "
            f"Total truncations: {self._output_truncations}.",
            flush=True,
        )


class BaseEmbedder(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts. Returns (len(texts), dim) array."""
        ...

    @abstractmethod
    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...

    @property
    def max_tokens(self) -> int:
        """Max input tokens the model supports. Override in subclasses."""
        return getattr(self, "_max_tokens", 8192)
