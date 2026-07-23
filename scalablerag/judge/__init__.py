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

from .evaluation import (
    ANSWER_ONLY_RETRY_PROMPT,
    ANSWER_ONLY_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    extract_answer_only,
    judge_one,
)
from .squad_metrics import best_squad_scores, normalize_answer, squad_em, squad_f1

__all__ = [
    "ANSWER_ONLY_RETRY_PROMPT",
    "ANSWER_ONLY_SYSTEM_PROMPT",
    "JUDGE_SYSTEM_PROMPT",
    "best_squad_scores",
    "extract_answer_only",
    "judge_one",
    "normalize_answer",
    "squad_em",
    "squad_f1",
]
