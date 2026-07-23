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

"""Answer extraction and scoring for batch evaluation."""

from __future__ import annotations

import json
from typing import Any

ANSWER_ONLY_SYSTEM_PROMPT = """You are a QA assistant.

You will be given:
- the question
- a model answer that may contain reasoning

Your task:
- Output ONLY the final answer to the question.
- Do NOT include any reasoning, explanation, preamble, or extra text.
- Do NOT use markdown code fences.

If the answer cannot be determined from the provided model answer, output exactly:
INSUFFICIENT_EVIDENCE
"""

ANSWER_ONLY_RETRY_PROMPT = """If the model answer contains any explicit final answer, you MUST output it.
Copy the final answer text as-is (verbatim) from the model answer when possible.
Only output INSUFFICIENT_EVIDENCE if the model answer truly contains no explicit final answer."""


def _parse_first_int(text: str) -> int | None:
    cur: list[str] = []
    seen_digit = False
    for ch in (text or ""):
        if ch.isdigit() or (ch == "-" and not seen_digit and not cur):
            cur.append(ch)
            if ch.isdigit():
                seen_digit = True
        else:
            if seen_digit:
                break
            cur = []
            seen_digit = False
    if not seen_digit:
        return None
    try:
        return int("".join(cur))
    except Exception:
        return None


def extract_answer_only(llm: "Any", question: str, model_answer: str) -> str:
    def _build_prompt(extra: str = "") -> str:
        extra_s = ("\n\n" + extra.strip()) if extra.strip() else ""
        return f"""{ANSWER_ONLY_SYSTEM_PROMPT}{extra_s}

Question: {question}
Model answer: {model_answer}

Final answer:"""

    prompt = _build_prompt()

    ctx_len = int(
        getattr(llm, "context_length", None)
        or getattr(llm, "_context_length", None)
        or 1_000_000
    )
    prompt_tokens = llm.count_tokens(prompt)
    available = ctx_len - int(prompt_tokens) - 32
    safe_mt = max(64, min(256, available))
    if available < 64:
        print(
            f"[extract_answer_only] WARNING: only {available} tokens available for extractor "
            f"completion (prompt={prompt_tokens}, ctx={ctx_len}). "
            f"Extractor output may be truncated.",
            flush=True,
        )

    txt = (llm.complete(prompt, max_tokens=safe_mt) or "").strip()
    if txt.startswith("```"):
        first_nl = txt.find("\n")
        if first_nl != -1:
            txt = txt[first_nl + 1 :]
        end = txt.rfind("```")
        if end != -1:
            txt = txt[:end]
        txt = txt.strip()
    if "```" in txt:
        txt = txt.split("```", 1)[0].strip()

    # Do NOT truncate to first line: the extractor may legitimately return
    # multi-line answers (including multi-line JSON). Downstream judging treats
    # this as opaque text.
    out = txt.strip()
    if out == "INSUFFICIENT_EVIDENCE" and (model_answer or "").strip():
        # Retry once with a stronger instruction; local and API models
        # sometimes overuse INSUFFICIENT_EVIDENCE.
        prompt2 = _build_prompt(ANSWER_ONLY_RETRY_PROMPT)
        out2 = (llm.complete(prompt2, max_tokens=safe_mt) or "").strip()
        if out2.startswith("```"):
            first_nl = out2.find("\n")
            if first_nl != -1:
                out2 = out2[first_nl + 1 :]
            end = out2.rfind("```")
            if end != -1:
                out2 = out2[:end]
            out2 = out2.strip()
        if "```" in out2:
            out2 = out2.split("```", 1)[0].strip()
        out = out2.strip() or out
    return out


JUDGE_SYSTEM_PROMPT = """You are an accurate evaluator for QA.
You will be given:
- the question
- the gold answer
- the model answer (final answer only; no reasoning)

Decide if the model answer matches the gold answer.
Be strict about factual correctness, but allow paraphrases and equivalent numeric formats.
If the gold answer is empty, mark as incorrect unless the model explicitly says it cannot be determined (e.g., INSUFFICIENT_EVIDENCE).

INSUFFICIENT_EVIDENCE rule:
- If the model answer is exactly the string "INSUFFICIENT_EVIDENCE" AND the gold answer is non-empty, verdict MUST be "incorrect" and score MUST be 0.

Special scoring rules:

1) Count questions:
- If the question is asking for a count/number, the answer is ONLY correct if the predicted number exactly matches the gold number.
- If it is off by even 1, score MUST be 0 and verdict MUST be "incorrect".

2) List/set questions:
- Treat the gold answer and model answer as sets of items.
- Let L be the number of distinct gold items.
- Start with score=1.0.
- Apply a penalty of (1/L) for each missing gold item AND for each extra predicted item.
- Treat possible aliases as correct. Do not give partial score per element.
- Score is floored at 0.
- If the model includes additional explanatory text, ignore it and focus on the item set.

When the answer is a proper name / named entity:
- Treat clear aliases or alternative spellings as correct (score=1).
- If the model answer is ambiguous (could refer to multiple entities) but not clearly wrong, allow partial credit with a short rationale.
- Give score=0 only if it is clearly the wrong entity.

Return a JSON object with:
- verdict: one of ["correct","incorrect","partial"]
- score: number in [0,1] where correct=1, incorrect=0, partial is between
- rationale: short explanation (1-3 sentences)
- abs_diff: OPTIONAL. Include ONLY for count/number questions. It must be a non-negative integer equal to |gold_number - predicted_number| if you can extract both numbers; otherwise null.

Rules:
- Output must be valid JSON (no markdown fences, no extra text).
"""


def judge_one(llm: "Any", question: str, gold: str, answer: str) -> tuple[str, float, str, int | None]:
    """Use LLM as judge to evaluate answer. Returns (verdict, score, rationale).

    ``llm`` must have a ``.complete(prompt, max_tokens=...)`` method (any
    provider: OpenAI, local, etc.).
    """
    prompt = f"""{JUDGE_SYSTEM_PROMPT}

Question: {question}
Gold answer: {gold}
Model answer: {answer}

JSON:"""

    ctx_len = int(
        getattr(llm, "context_length", None)
        or getattr(llm, "_context_length", None)
        or 1_000_000
    )
    prompt_tokens = llm.count_tokens(prompt)
    available = ctx_len - int(prompt_tokens) - 32
    safe_mt = max(150, min(300, available))
    if available < 150:
        print(
            f"[judge_one] WARNING: only {available} tokens available for judge "
            f"completion (prompt={prompt_tokens}, ctx={ctx_len}). "
            f"Judge output may be truncated.",
            flush=True,
        )
    def _strip_fences(s: str) -> str:
        s = (s or "").strip()
        if s.startswith("```"):
            first_nl = s.find("\n")
            if first_nl != -1:
                s = s[first_nl + 1 :]
            end = s.rfind("```")
            if end != -1:
                s = s[:end]
        if "```" in s:
            s = s.split("```", 1)[0]
        return s.strip()

    def _complete_json(p: str) -> str:
        # Prefer JSON-mode if supported by the provider/server. If unsupported,
        # fall back to plain completion (still instructed to output JSON only).
        try:
            return llm.complete(p, max_tokens=safe_mt, response_format={"type": "json_object"})
        except TypeError:
            return llm.complete(p, max_tokens=safe_mt)
        except Exception:
            # Some servers reject response_format; retry without it.
            try:
                return llm.complete(p, max_tokens=safe_mt)
            except Exception:
                raise

    last_txt = ""
    for attempt in range(3):
        resp = _complete_json(prompt if attempt == 0 else f"""{prompt}

Your previous output was not valid JSON.
Previous output:
{last_txt}

Re-output JSON only:""")
        txt = _strip_fences((resp or "").strip())
        last_txt = txt
        try:
            obj = json.loads(txt)
            verdict = str(obj.get("verdict", "incorrect"))
            score = float(obj.get("score", 0.0))
            rationale = str(obj.get("rationale", ""))
            abs_diff_raw = obj.get("abs_diff", None)
            abs_diff: int | None = None
            if abs_diff_raw is None:
                abs_diff = None
            elif isinstance(abs_diff_raw, bool):
                abs_diff = None
            elif isinstance(abs_diff_raw, (int, float)) and float(abs_diff_raw) >= 0:
                abs_diff = int(abs_diff_raw)
            elif isinstance(abs_diff_raw, str):
                s = abs_diff_raw.strip()
                if s.isdigit():
                    abs_diff = int(s)
            return verdict, score, rationale, abs_diff
        except Exception:
            continue
    raise ValueError(f"judge did not return valid JSON after retries: {last_txt[:200]}...")
