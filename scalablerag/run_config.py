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

"""Run flag resolution and embedding setup for eval and query."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scalablerag.llm.factory import make_embedder
from scalablerag.storage.embedding_store import EmbeddingStore


REGIME_PRESETS: dict[str, dict[str, Any]] = {
    "zero": {
        "skip_embeddings": True,
        "discover_patterns": False,
        "curate_patterns": False,
        "use_agg_tools": True,
        "branch_select": True,
        "require_embeddings": False,
    },
    "limited": {
        "skip_embeddings": False,
        "discover_patterns": True,
        "curate_patterns": True,
        "use_agg_tools": True,
        "branch_select": True,
        "require_embeddings": True,
        "embed_provider": "openai",
        "embed_model": "text-embedding-3-small",
    },
}

DEFAULT_RUN = {
    "skip_embeddings": False,
    "discover_patterns": False,
    "curate_patterns": False,
    "use_agg_tools": True,
    "branch_select": True,
    "require_embeddings": False,
}

CLI_FLAG_GROUPS: dict[str, tuple[str, ...]] = {
    "embeddings": ("--embeddings", "--no-embeddings"),
    "discover_patterns": ("--discover-patterns", "--no-discover-patterns"),
    "curate_patterns": ("--curate-patterns", "--no-curate-patterns"),
    "agg_tools": ("--agg-tools", "--no-agg-tools"),
    "branch_select": ("--branch-select", "--no-branch-select"),
}


def detect_explicit_flags(argv: list[str]) -> set[str]:
    """Flag groups present on argv."""
    explicit: set[str] = set()
    for group, names in CLI_FLAG_GROUPS.items():
        if any(n in argv for n in names):
            explicit.add(group)
    return explicit


@dataclass
class ResolvedRunConfig:
    regime: str | None
    skip_embeddings: bool
    discover_patterns: bool
    curate_patterns: bool
    use_agg_tools: bool
    branch_select: bool
    require_embeddings: bool
    embed_provider: str | None
    embed_model: str | None
    embed_dir: str


def load_embedding_setup(
    store_path: Path,
    *,
    skip_embeddings: bool,
    embed_provider: str | None = None,
    embed_model: str | None = None,
    embed_dir: str = "",
) -> tuple[EmbeddingStore | None, object | None, bool]:
    if skip_embeddings:
        return None, None, False

    embedder = make_embedder(provider=embed_provider, model=embed_model)
    embedding_store = EmbeddingStore.find_compatible_store(
        store_path, embedder.dimension, embed_dir=embed_dir,
    )
    return embedding_store, embedder, embedding_store is not None


def _env_bool(name: str) -> bool | None:
    raw = os.environ.get(name, "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return None


def resolve_run_config(
    args: Any,
    explicit: set[str] | None = None,
) -> ResolvedRunConfig:
    """Merge --regime, CLI flags, and SR_* env."""
    explicit = explicit or set()
    preset = dict(REGIME_PRESETS.get(getattr(args, "regime", None) or "", {}))
    base = dict(DEFAULT_RUN)
    base.update(preset)

    if "embeddings" in explicit:
        skip_embeddings = not bool(getattr(args, "embeddings", True))
    elif _env_bool("SR_NO_EMBEDDINGS") is True:
        skip_embeddings = True
    else:
        skip_embeddings = bool(base.get("skip_embeddings", False))

    def _resolve_bool(group: str, attr: str, preset_key: str, env_key: str) -> bool:
        if group in explicit:
            return bool(getattr(args, attr, False))
        env = _env_bool(env_key)
        if env is not None:
            return env
        return bool(base.get(preset_key, DEFAULT_RUN.get(preset_key, False)))

    discover = _resolve_bool(
        "discover_patterns", "discover_patterns", "discover_patterns", "SR_DISCOVER_PATTERNS"
    )
    curate = _resolve_bool(
        "curate_patterns", "curate_patterns", "curate_patterns", "SR_CURATE_PATTERNS"
    )
    use_agg = _resolve_bool("agg_tools", "agg_tools", "use_agg_tools", "SR_AGG_TOOLS")
    branch = _resolve_bool(
        "branch_select", "branch_select", "branch_select", "SR_BRANCH_SELECT"
    )

    require_emb = bool(base.get("require_embeddings", False))
    if _env_bool("SR_REQUIRE_EMBEDDINGS") is True:
        require_emb = True
    if "embeddings" in explicit and bool(getattr(args, "embeddings", False)):
        require_emb = True
    if skip_embeddings:
        require_emb = False

    embed_provider = getattr(args, "embed_provider", None) or base.get("embed_provider")
    embed_model = getattr(args, "embed_model", None) or base.get("embed_model")
    embed_dir = getattr(args, "embed_dir", None) or os.environ.get("SR_EMBED_DIR", "")

    return ResolvedRunConfig(
        regime=getattr(args, "regime", None),
        skip_embeddings=skip_embeddings,
        discover_patterns=discover,
        curate_patterns=curate,
        use_agg_tools=use_agg,
        branch_select=branch,
        require_embeddings=require_emb,
        embed_provider=embed_provider,
        embed_model=embed_model,
        embed_dir=embed_dir or "",
    )
