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

"""Bounded execution for user-supplied regex patterns.

Static pattern checks run in-process.  Bulk matching runs in a single
subprocess with a wall-clock timeout so catastrophic backtracking cannot
hang the agent loop.  Full document text is scanned — nothing is truncated.
"""

from __future__ import annotations

import multiprocessing
import re
from typing import Any

REGEX_OP_TIMEOUT = 30.0
MAX_REGEX_CHARS = 500

_NESTED_QUANTIFIERS = re.compile(
    r"\([^)]*[+*][^)]*\)[+*?]"
    r"|"
    r"\([^)]*\([^)]*[+*][^)]*\)[^)]*\)[+*?]"
)

TIMEOUT_ERROR = (
    "Error: regex operation timed out after 30 seconds. "
    "Simplify the pattern or narrow the target set."
)


class RegexTimeoutError(Exception):
    """Raised when a regex batch exceeds the wall-clock budget."""


class RegexSafetyError(Exception):
    """Raised when a user pattern fails static safety checks."""


def validate_user_pattern(pattern: str) -> None:
    """Raise RegexSafetyError if the pattern looks unsafe or invalid."""
    if not pattern:
        raise RegexSafetyError("empty pattern")
    if len(pattern) > MAX_REGEX_CHARS:
        raise RegexSafetyError(
            f"pattern exceeds {MAX_REGEX_CHARS} characters "
            f"(got {len(pattern)})"
        )
    try:
        re.compile(pattern)
    except re.error as exc:
        raise RegexSafetyError(f"invalid regex: {exc}") from exc
    if _NESTED_QUANTIFIERS.search(pattern):
        raise RegexSafetyError(
            "pattern contains nested quantifiers that may cause excessive backtracking"
        )


def compile_user_pattern(
    pattern: str,
    flags: int = re.IGNORECASE,
) -> re.Pattern[str]:
    """Validate and compile a user-supplied pattern."""
    validate_user_pattern(pattern)
    return re.compile(pattern, flags)


def _run_timed(worker: Any, args: tuple[Any, ...], timeout: float) -> Any:
    ctx = multiprocessing.get_context("fork")
    pool = ctx.Pool(1)
    try:
        async_result = pool.apply_async(worker, args)
        return async_result.get(timeout=timeout)
    except multiprocessing.TimeoutError as exc:
        pool.terminate()
        raise RegexTimeoutError(TIMEOUT_ERROR) from exc
    except Exception:
        pool.terminate()
        raise
    finally:
        pool.close()
        pool.join()


def _worker_search_many(
    pattern_str: str, flags: int, texts: list[str]
) -> list[bool]:
    rx = re.compile(pattern_str, flags)
    return [rx.search(t) is not None for t in texts]


def _worker_multi_search(
    pattern_strs: list[str], flags: int, texts: list[str]
) -> list[list[bool]]:
    compiled = [re.compile(p, flags) for p in pattern_strs]
    rows: list[list[bool]] = []
    for rx in compiled:
        rows.append([rx.search(t) is not None for t in texts])
    return rows


def _worker_search_group(
    pattern_str: str,
    flags: int,
    texts: list[str],
    group: int | str,
) -> list[str | None]:
    rx = re.compile(pattern_str, flags)
    out: list[str | None] = []
    for t in texts:
        m = rx.search(t)
        if not m:
            out.append(None)
            continue
        try:
            out.append(m.group(group))
        except (IndexError, re.error):
            out.append(None)
    return out


def _worker_findall_groups(
    pattern_str: str,
    flags: int,
    texts: list[str],
    group: int | str,
) -> list[list[str]]:
    rx = re.compile(pattern_str, flags)
    all_docs: list[list[str]] = []
    for t in texts:
        matches = rx.findall(t)
        items: list[str] = []
        for m in matches:
            if isinstance(m, tuple):
                if isinstance(group, int) and group > 0:
                    raw = m[group - 1]
                else:
                    raw = m[0] if m else ""
            else:
                raw = m
            val = raw.strip() if isinstance(raw, str) else str(raw).strip()
            if val:
                items.append(val)
        all_docs.append(items)
    return all_docs


def _worker_finditer_spans(
    pattern_str: str,
    flags: int,
    text: str,
    max_matches: int,
) -> list[tuple[int, int]]:
    rx = re.compile(pattern_str, flags)
    spans: list[tuple[int, int]] = []
    for m in rx.finditer(text):
        spans.append((m.start(), m.end()))
        if max_matches > 0 and len(spans) >= max_matches:
            break
    return spans


def search_many(
    pattern: str,
    flags: int,
    texts: list[str],
    *,
    timeout: float = REGEX_OP_TIMEOUT,
) -> list[bool]:
    """Search each text; entire batch must finish within *timeout* seconds."""
    compile_user_pattern(pattern, flags)
    return _run_timed(_worker_search_many, (pattern, flags, texts), timeout)


def search_multi(
    patterns: list[str],
    flags: int,
    texts: list[str],
    *,
    timeout: float = REGEX_OP_TIMEOUT,
) -> list[list[bool]]:
    """Search all patterns against all texts under one timeout."""
    for pat in patterns:
        validate_user_pattern(pat)
    return _run_timed(_worker_multi_search, (patterns, flags, texts), timeout)


def search_group_many(
    pattern: str,
    flags: int,
    texts: list[str],
    group: int | str,
    *,
    timeout: float = REGEX_OP_TIMEOUT,
) -> list[str | None]:
    """Return the capture group (or None) for the first match in each text."""
    compile_user_pattern(pattern, flags)
    return _run_timed(
        _worker_search_group, (pattern, flags, texts, group), timeout
    )


def findall_groups_many(
    pattern: str,
    flags: int,
    texts: list[str],
    group: int | str,
    *,
    timeout: float = REGEX_OP_TIMEOUT,
) -> list[list[str]]:
    """Return extracted capture groups for every match in each text."""
    compile_user_pattern(pattern, flags)
    return _run_timed(
        _worker_findall_groups, (pattern, flags, texts, group), timeout
    )


def finditer_spans(
    pattern: str,
    flags: int,
    text: str,
    *,
    max_matches: int = 0,
    timeout: float = REGEX_OP_TIMEOUT,
) -> list[tuple[int, int]]:
    """Return match spans for one text under a timeout.

    *max_matches* <= 0 means no limit (same as unbounded finditer).
    """
    compile_user_pattern(pattern, flags)
    return _run_timed(
        _worker_finditer_spans,
        (pattern, flags, text, max_matches),
        timeout,
    )


def format_safety_error(exc: RegexSafetyError) -> str:
    return f"Error: regex pattern rejected: {exc}"
