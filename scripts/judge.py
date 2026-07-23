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

"""Score eval JSONL with answer extraction and LLM verdicts."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scalablerag.judge.evaluation import (  # noqa: E402
    _parse_first_int,
    extract_answer_only,
    judge_one,
)
from scalablerag.judge.squad_metrics import best_squad_scores  # noqa: E402
from scalablerag.llm.base import BaseLLM  # noqa: E402
from scalablerag.llm.factory import KNOWN_LLM_PROVIDERS, make_llm  # noqa: E402


class _JudgeLLMAdapter:
    """Adapt ScalableRAG BaseLLM to the interface used by judge evaluation."""

    def __init__(self, llm: BaseLLM) -> None:
        self._llm = llm

    @property
    def context_length(self) -> int:
        return self._llm.context_length

    def count_tokens(self, text: str) -> int:
        return self._llm.count_tokens(text)

    def complete(
        self,
        prompt: str,
        max_tokens: int = 256,
        *,
        response_format: dict[str, str] | None = None,
    ) -> str:
        json_schema = {} if response_format else None
        return self._llm.complete(
            prompt, max_tokens=max_tokens, json_schema=json_schema
        )


def _iter_jsonl(path: str) -> Iterable[dict[str, Any]]:
    if path == "-":
        for line in sys.stdin:
            if line.strip():
                yield json.loads(line)
        return
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _get_str(obj: dict[str, Any], keys: list[str], default: str = "") -> str:
    for k in keys:
        v = obj.get(k)
        if v is None:
            continue
        s = str(v)
        if s.strip():
            return s
    return default


def _get_aliases(obj: dict[str, Any]) -> list[str]:
    raw = obj.get("answer_aliases")
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw.strip()] if raw.strip() else []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-jsonl", required=True, help="Input JSONL (use '-' for stdin)")
    ap.add_argument(
        "--out-dir", required=True, help="Output directory (writes judged JSONL + summary JSON)"
    )
    ap.add_argument(
        "--llm-provider",
        default=None,
        choices=list(KNOWN_LLM_PROVIDERS),
        help="LLM provider (default: SR_LLM_PROVIDER env or openai)",
    )
    ap.add_argument(
        "--llm-model",
        default=None,
        help="LLM model or deployment name (default: SR_LLM_MODEL env or gpt-4.1)",
    )
    ap.add_argument(
        "--log-every",
        type=int,
        default=int(os.environ.get("SR_LOG_EVERY", "1") or "1"),
        help="Print one progress line every N questions (default: 1)",
    )
    args = ap.parse_args()

    out_dir = str(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    in_base = "stdin" if str(args.in_jsonl) == "-" else os.path.basename(str(args.in_jsonl))
    if in_base.endswith(".jsonl"):
        in_base = in_base[: -len(".jsonl")]
    out_jsonl_path = os.path.join(out_dir, f"{in_base}_judged.jsonl")
    out_summary_path = os.path.join(out_dir, f"summary_{in_base}_judged.json")

    llm = _JudgeLLMAdapter(
        make_llm(
            provider=args.llm_provider,
            repo_root=_REPO_ROOT,
            model=args.llm_model,
        )
    )

    n = 0
    score_sum = 0.0
    em_sum = 0.0
    f1_sum = 0.0
    t0 = time.time()
    verdict_counts: dict[str, int] = {"correct": 0, "incorrect": 0, "partial": 0}
    type_scores: dict[str, list[float]] = {}
    type_em: dict[str, list[float]] = {}
    type_f1: dict[str, list[float]] = {}
    count_abs_diffs: list[int] = []
    n_count_with_abs_diff = 0
    with open(out_jsonl_path, "w", encoding="utf-8") as f_out:
        for row in _iter_jsonl(str(args.in_jsonl)):
            t_q = time.time()
            qid = _get_str(row, ["question_id", "qid", "id"], default=str(n + 1)).strip() or str(
                n + 1
            )
            question = _get_str(row, ["question", "q", "query", "query_text"])
            gold = _get_str(row, ["golden_answer", "gold", "gold_answer"])
            model_answer = _get_str(
                row, ["answer", "answer_raw", "model_answer", "prediction", "pred"]
            )
            qtype = _get_str(row, ["question_type", "type"], default="unknown").strip() or "unknown"

            extracted = extract_answer_only(llm, question, model_answer)
            if not extracted:
                extracted = model_answer.strip()

            verdict, score, rationale, abs_diff = judge_one(llm, question, gold, extracted)
            em, f1 = best_squad_scores(extracted, gold, _get_aliases(row))
            # Do not allow the judge to credit "INSUFFICIENT_EVIDENCE" when the gold answer is non-empty.
            if extracted.strip() == "INSUFFICIENT_EVIDENCE":
                if (gold or "").strip():
                    verdict = "incorrect"
                    score = 0.0
                    rationale = "Model answered INSUFFICIENT_EVIDENCE but gold answer is non-empty."
                else:
                    verdict = "correct"
                    score = 1.0
                    rationale = "Gold answer is empty and model answered INSUFFICIENT_EVIDENCE."
            judge_payload: dict[str, Any] = {
                "verdict": verdict,
                "score": score,
                "rationale": rationale,
                "squad_em": em,
                "squad_f1": f1,
            }
            if "count" in qtype.lower():
                # Do NOT trust the judge to follow the abs_diff contract.
                # Compute deterministically from the extracted strings when possible.
                gold_i = _parse_first_int(gold)
                pred_i = _parse_first_int(extracted)
                abs_diff_computed: int | None = None
                if gold_i is not None and pred_i is not None:
                    abs_diff_computed = abs(gold_i - pred_i)
                judge_payload["abs_diff"] = abs_diff_computed
                judge_payload["abs_diff_llm"] = abs_diff
                if isinstance(abs_diff_computed, int):
                    count_abs_diffs.append(int(abs_diff_computed))
                    n_count_with_abs_diff += 1

            verdict = str(judge_payload.get("verdict", "incorrect"))
            score = float(judge_payload.get("score", 0.0))
            verdict_counts[verdict] = int(verdict_counts.get(verdict, 0)) + 1
            type_scores.setdefault(qtype, []).append(score)
            type_em.setdefault(qtype, []).append(em)
            type_f1.setdefault(qtype, []).append(f1)

            row["judge"] = judge_payload
            row["answer_extracted"] = extracted
            row["answer_model_raw"] = model_answer

            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            f_out.flush()

            n += 1
            score_sum += float(score)
            em_sum += float(em)
            f1_sum += float(f1)
            elapsed = round(time.time() - t_q, 1)
            if int(args.log_every) > 0 and (n % int(args.log_every) == 0 or n == 1):
                avg = score_sum / max(1, n)
                print(
                    f"[{n}] qid={qid} type={qtype} verdict={verdict} score={score:.2f} "
                    f"em={em:.0f} f1={f1:.2f} t={elapsed:.1f}s running_avg={avg:.4f}",
                    flush=True,
                )

    total = time.time() - t0
    avg = score_sum / max(1, n)
    avg_em = em_sum / max(1, n)
    avg_f1 = f1_sum / max(1, n)
    avg_abs_diff = (sum(count_abs_diffs) / len(count_abs_diffs)) if count_abs_diffs else None
    summary = {
        "avg_score": avg,
        "squad_em_pct": round(100.0 * avg_em, 4),
        "squad_f1_pct": round(100.0 * avg_f1, 4),
        "verdict_counts": verdict_counts,
        "per_type": {
            qt: {
                "n": len(vals),
                "avg": round(sum(vals) / len(vals), 4),
                "squad_em_pct": round(
                    100.0 * sum(type_em.get(qt, [])) / max(1, len(type_em.get(qt, []))), 4
                ),
                "squad_f1_pct": round(
                    100.0 * sum(type_f1.get(qt, [])) / max(1, len(type_f1.get(qt, []))), 4
                ),
            }
            for qt, vals in sorted(type_scores.items())
        },
        "count_questions": {
            "n_count_with_abs_diff": int(n_count_with_abs_diff),
            "n_count_abs_diff_values": int(len(count_abs_diffs)),
            "avg_abs_diff": (None if avg_abs_diff is None else float(avg_abs_diff)),
        },
        "total_seconds": round(total, 1),
        "output_jsonl": out_jsonl_path,
    }
    with open(out_summary_path, "w", encoding="utf-8") as f_sum:
        f_sum.write(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}", flush=True)
    print(
        f"DONE  n={n}  avg={avg:.4f}  em={avg_em:.4f}  f1={avg_f1:.4f}  "
        f"counts={verdict_counts}  total_time={total:.0f}s",
        flush=True,
    )
    if type_scores:
        items = [(qt, vals) for qt, vals in type_scores.items() if vals]
        items.sort(key=lambda kv: -(sum(kv[1]) / len(kv[1])))
        for qt, vals in items:
            print(f"  {qt:<22} n={len(vals):>3}  avg={sum(vals)/len(vals):.4f}", flush=True)
    if count_abs_diffs:
        print(
            f"Count abs_diff: n_values={len(count_abs_diffs)} avg_abs_diff={avg_abs_diff}",
            flush=True,
        )
    print(f"Wrote {out_jsonl_path}", flush=True)
    print(f"Wrote {out_summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
