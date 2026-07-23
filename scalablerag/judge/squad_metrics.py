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

"""SQuAD-style exact match and token-overlap F1.

Used on ``answer_extracted`` (post LLM normalization) vs ``golden_answer``,
taking the max score over ``answer_aliases`` when present.
"""

from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation/articles, collapse whitespace."""
    s = (text or "").lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def squad_em(prediction: str, gold: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def squad_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def gold_variants(gold: str, aliases: list[str] | None = None) -> list[str]:
    gts: list[str] = [gold if gold is not None else ""]
    if isinstance(aliases, list):
        for alias in aliases:
            if isinstance(alias, str) and alias.strip():
                gts.append(alias)
    out = [
        x
        for x in dict.fromkeys(gt.strip() for gt in gts if gt is not None)
        if x != ""
    ]
    return out or [""]


def best_squad_scores(
    prediction: str, gold: str, aliases: list[str] | None = None
) -> tuple[float, float]:
    """Return max EM/F1 over gold answer and aliases."""
    em = 0.0
    f1 = 0.0
    for variant in gold_variants(gold, aliases):
        em = max(em, squad_em(prediction, variant))
        f1 = max(f1, squad_f1(prediction, variant))
    return em, f1
